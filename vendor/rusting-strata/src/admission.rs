//! Bounded admission for generation requests, independently of HTTP connection handling.
//!
//! A ticket reserves a running slot as soon as one is available, in FIFO order.
//! `poll_wait` claims that reservation as an owned permit; keep the permit with the
//! request until inference (including response cleanup) ends. Unclaimed tickets
//! remain subject to their original enqueue deadline, even while reserved.
//! Control/health requests should never enter this controller.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    Full,
    Timeout,
    Cancelled,
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Full => "the generation queue is full",
            Self::Timeout => "the generation request timed out while waiting for a slot",
            Self::Cancelled => "the generation request was cancelled or its permit was already claimed",
        })
    }
}

impl std::error::Error for AdmissionError {}

/// `active` counts owned permits; `reserved` counts tickets offered a slot but
/// not yet claimed by their HTTP reader. Both consume running capacity.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Snapshot {
    pub active: usize,
    pub reserved: usize,
    pub queued: usize,
    pub capacity: usize,
    pub queue_capacity: usize,
    pub queue_timeout_seconds: f64,
    pub rejected_full_total: u64,
    pub timed_out_total: u64,
    pub cancelled_total: u64,
}

#[derive(Clone)]
pub struct Admission {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    changed: Condvar,
    wait_timeout: Duration,
}

struct State {
    capacity: usize,
    queue_capacity: usize,
    active: usize,
    reserved: usize,
    /// Both waiting and reserved tickets, retaining their original FIFO order.
    /// Terminal tickets are held only by their caller, never by this queue.
    pending: VecDeque<Arc<Pending>>,
    rejected_full_total: u64,
    timed_out_total: u64,
    cancelled_total: u64,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Status { Waiting, Reserved, Claimed, TimedOut, Cancelled }

struct Pending {
    enqueued: Instant,
    // checked_add can overflow only for an effectively unbounded Duration.
    deadline: Option<Instant>,
    // All accesses are serialized by Inner::state. An atomic lets a ticket
    // retain its terminal result after the queue removes its Arc, with no
    // second mutex or unsafe shared mutation.
    status: AtomicU8,
}

impl Pending {
    fn status(&self) -> Status {
        match self.status.load(Ordering::Relaxed) {
            0 => Status::Waiting,
            1 => Status::Reserved,
            2 => Status::Claimed,
            3 => Status::TimedOut,
            4 => Status::Cancelled,
            _ => unreachable!("invalid admission ticket status"),
        }
    }

    fn set_status(&self, status: Status) { self.status.store(status as u8, Ordering::Relaxed); }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl State {
    /// Expiry precedes promotion, including expiry at the exact deadline.
    /// Recompute reservations after a shrink rather than letting an unclaimed
    /// ticket start against unavailable capacity. Already-owned permits drain.
    fn maintain(&mut self, now: Instant) -> bool {
        let mut expired = 0u64;
        self.pending.retain(|ticket| {
            if ticket.deadline.is_some_and(|deadline| now >= deadline) {
                ticket.set_status(Status::TimedOut);
                expired = expired.saturating_add(1);
                false
            } else { true }
        });
        self.timed_out_total = self.timed_out_total.saturating_add(expired);
        let available = self.capacity.saturating_sub(self.active);
        let mut changed = expired > 0;
        for (index, ticket) in self.pending.iter().enumerate() {
            let status = if index < available { Status::Reserved } else { Status::Waiting };
            if ticket.status() != status {
                ticket.set_status(status);
                changed = true;
            }
        }
        self.reserved = self.pending.len().min(available);
        changed
    }

    fn queued(&self) -> usize { self.pending.len() - self.reserved }

    fn remove(&mut self, ticket: &Arc<Pending>) {
        let index = self.pending.iter().position(|p| Arc::ptr_eq(p, ticket))
            .expect("live admission ticket missing from its queue");
        self.pending.remove(index);
    }
}

impl Admission {
    /// `capacity` may be zero while no model can execute requests.
    /// `queue_capacity == 0` accepts only requests with immediately free slots.
    /// A zero wait timeout expires every unclaimed ticket immediately; callers
    /// wanting immediate-only admission should instead set queue capacity to 0.
    pub fn new(capacity: usize, queue_capacity: usize, wait_timeout: Duration) -> Self {
        Self { inner: Arc::new(Inner {
            state: Mutex::new(State {
                capacity, queue_capacity, active: 0, reserved: 0,
                pending: VecDeque::new(), rejected_full_total: 0,
                timed_out_total: 0, cancelled_total: 0,
            }),
            changed: Condvar::new(), wait_timeout,
        }) }
    }

    /// Does not wait. A successful enqueue owns either a running reservation or
    /// one bounded waiting position; the deadline starts at this enqueue.
    pub fn enqueue(&self) -> Result<Ticket, AdmissionError> {
        let mut state = self.inner.lock();
        self.enqueue_locked(&mut state, Instant::now())
    }

    fn enqueue_locked(&self, state: &mut State, now: Instant) -> Result<Ticket, AdmissionError> {
        let changed = state.maintain(now);
        let available = state.active.saturating_add(state.reserved) < state.capacity;
        if !available && state.queued() >= state.queue_capacity {
            state.rejected_full_total = state.rejected_full_total.saturating_add(1);
            if changed { self.inner.changed.notify_all(); }
            return Err(AdmissionError::Full);
        }
        let pending = Arc::new(Pending {
            enqueued: now, deadline: now.checked_add(self.inner.wait_timeout),
            status: AtomicU8::new(Status::Waiting as u8),
        });
        state.pending.push_back(pending.clone());
        state.maintain(now);
        self.inner.changed.notify_all();
        if pending.status() == Status::TimedOut { return Err(AdmissionError::Timeout); }
        Ok(Ticket { inner: self.inner.clone(), pending })
    }

    /// Shrinking never revokes an owned permit. Excess unclaimed reservations
    /// return to their original FIFO waiting positions. Those already accepted
    /// tickets may temporarily exceed queue_capacity after a shrink; no new
    /// arrival can use that exception, and their deadlines do not restart.
    pub fn set_capacity(&self, capacity: usize) {
        let mut state = self.inner.lock();
        state.capacity = capacity;
        state.maintain(Instant::now());
        self.inner.changed.notify_all();
    }

    pub fn snapshot(&self) -> Snapshot {
        let mut state = self.inner.lock();
        self.snapshot_locked(&mut state, Instant::now())
    }

    fn snapshot_locked(&self, state: &mut State, now: Instant) -> Snapshot {
        if state.maintain(now) { self.inner.changed.notify_all(); }
        Snapshot {
            active: state.active, reserved: state.reserved, queued: state.queued(),
            capacity: state.capacity, queue_capacity: state.queue_capacity,
            queue_timeout_seconds: self.inner.wait_timeout.as_secs_f64(),
            rejected_full_total: state.rejected_full_total,
            timed_out_total: state.timed_out_total,
            cancelled_total: state.cancelled_total,
        }
    }

    #[cfg(test)]
    fn enqueue_at(&self, now: Instant) -> Result<Ticket, AdmissionError> {
        self.enqueue_locked(&mut self.inner.lock(), now)
    }

    #[cfg(test)]
    fn set_capacity_at(&self, capacity: usize, now: Instant) {
        let mut state = self.inner.lock();
        state.capacity = capacity;
        state.maintain(now);
        self.inner.changed.notify_all();
    }

    #[cfg(test)]
    fn snapshot_at(&self, now: Instant) -> Snapshot {
        self.snapshot_locked(&mut self.inner.lock(), now)
    }
}

/// Dropping an unclaimed ticket cancels it and immediately frees its waiting
/// position/reservation. A claimed ticket's Drop never releases its Permit.
pub struct Ticket {
    inner: Arc<Inner>,
    pending: Arc<Pending>,
}

impl Ticket {
    /// Wait at most max_wait for a permit. Ok(None) means still queued: the
    /// caller can check the HTTP socket, then poll again. A zero interval is a
    /// nonblocking poll. At most one Permit can be claimed from each Ticket.
    pub fn poll_wait(&mut self, max_wait: Duration) -> Result<Option<Permit>, AdmissionError> {
        let inner = self.inner.clone();
        let mut state = inner.lock();
        let poll_deadline = Instant::now().checked_add(max_wait);
        loop {
            let now = Instant::now();
            match self.poll_locked(&mut state, now)? {
                Some(permit) => return Ok(Some(permit)),
                None => {}
            }
            let remaining = poll_deadline.map_or(max_wait, |end| end.saturating_duration_since(now));
            if remaining.is_zero() { return Ok(None); }
            let until_expiry = self.pending.deadline.map_or(remaining, |end| end.saturating_duration_since(now));
            let wait = remaining.min(until_expiry);
            // maintain in poll_locked already expires deadlines <= now.
            let (next, _) = inner.changed.wait_timeout(state, wait).unwrap_or_else(|e| e.into_inner());
            state = next;
        }
    }

    fn poll_locked(&self, state: &mut State, now: Instant) -> Result<Option<Permit>, AdmissionError> {
        if state.maintain(now) { self.inner.changed.notify_all(); }
        match self.pending.status() {
            Status::Waiting => Ok(None),
            Status::Reserved => {
                state.remove(&self.pending);
                state.active += 1;
                self.pending.set_status(Status::Claimed);
                state.maintain(now);
                self.inner.changed.notify_all();
                Ok(Some(Permit { inner: self.inner.clone(), waited: now.saturating_duration_since(self.pending.enqueued), released: false }))
            }
            Status::TimedOut => Err(AdmissionError::Timeout),
            Status::Cancelled | Status::Claimed => Err(AdmissionError::Cancelled),
        }
    }

    /// Returns true only when this call cancels a live ticket. Expiry at or
    /// before this call wins over cancellation and is counted only as timeout.
    pub fn cancel(&mut self) -> bool {
        let inner = self.inner.clone();
        let mut state = inner.lock();
        self.cancel_locked(&mut state, Instant::now())
    }

    fn cancel_locked(&self, state: &mut State, now: Instant) -> bool {
        if state.maintain(now) { self.inner.changed.notify_all(); }
        if !matches!(self.pending.status(), Status::Waiting | Status::Reserved) { return false; }
        state.remove(&self.pending);
        self.pending.set_status(Status::Cancelled);
        state.cancelled_total = state.cancelled_total.saturating_add(1);
        state.maintain(now);
        self.inner.changed.notify_all();
        true
    }

    #[cfg(test)]
    fn poll_at(&self, now: Instant) -> Result<Option<Permit>, AdmissionError> {
        self.poll_locked(&mut self.inner.lock(), now)
    }

    #[cfg(test)]
    fn cancel_at(&self, now: Instant) -> bool {
        self.cancel_locked(&mut self.inner.lock(), now)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) { self.cancel(); }
}

/// Owns one active generation slot. Moving this through the request channel
/// keeps admission bounded until the handler drops it, including error paths.
pub struct Permit {
    inner: Arc<Inner>,
    waited: Duration,
    released: bool,
}

impl Permit {
    pub fn wait_duration(&self) -> Duration { self.waited }

    fn release_locked(&mut self, state: &mut State, now: Instant) {
        if self.released { return; }
        state.active = state.active.checked_sub(1).expect("admission permit released without an active slot");
        self.released = true;
        state.maintain(now);
        self.inner.changed.notify_all();
    }

    #[cfg(test)]
    fn release_at(&mut self, now: Instant) {
        let inner = self.inner.clone();
        self.release_locked(&mut inner.lock(), now);
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if self.released { return; }
        let inner = self.inner.clone();
        let mut state = inner.lock();
        self.release_locked(&mut state, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    const LIMIT: Duration = Duration::from_secs(120);

    fn claim(ticket: &Ticket, now: Instant) -> Permit { ticket.poll_at(now).unwrap().expect("ticket should have a slot") }

    #[test]
    fn waiting_capacity_excludes_active_and_reserved_slots() {
        let gate = Admission::new(2, 1, LIMIT);
        let now = Instant::now();
        let a = gate.enqueue_at(now).unwrap();
        let b = gate.enqueue_at(now).unwrap();
        let c = gate.enqueue_at(now).unwrap();
        assert!(matches!(gate.enqueue_at(now), Err(AdmissionError::Full)));
        let p = claim(&a, now);
        let s = gate.snapshot_at(now);
        assert_eq!((s.active, s.reserved, s.queued), (1, 1, 1));
        assert_eq!((s.capacity, s.queue_capacity, s.rejected_full_total), (2, 1, 1));
        assert_eq!(s.queue_timeout_seconds, 120.0);
        drop(a);
        assert_eq!(gate.snapshot_at(now).active, 1, "dropping a claimed ticket cannot release its permit");
        drop((p, b, c));
        let s = gate.snapshot();
        assert_eq!((s.active, s.reserved, s.queued), (0, 0, 0));
        assert_eq!(s.cancelled_total, 2);
    }

    #[test]
    fn fifo_new_arrivals_cannot_steal_an_older_reservation() {
        let gate = Admission::new(0, 3, LIMIT);
        let now = Instant::now();
        let first = gate.enqueue_at(now).unwrap();
        let second = gate.enqueue_at(now).unwrap();
        gate.set_capacity_at(1, now);
        let third = gate.enqueue_at(now).unwrap();
        assert!(second.poll_at(now).unwrap().is_none());
        assert!(third.poll_at(now).unwrap().is_none());
        let mut p = claim(&first, now);
        p.release_at(now);
        assert!(third.poll_at(now).unwrap().is_none());
        let mut p = claim(&second, now);
        p.release_at(now);
        assert!(third.poll_at(now).unwrap().is_some());
    }

    #[test]
    fn zero_queue_capacity_is_immediate_only() {
        let gate = Admission::new(1, 0, LIMIT);
        let now = Instant::now();
        let ticket = gate.enqueue_at(now).unwrap();
        assert!(matches!(gate.enqueue_at(now), Err(AdmissionError::Full)));
        let mut p = claim(&ticket, now);
        p.release_at(now);
        assert!(gate.enqueue_at(now).is_ok());
        gate.set_capacity_at(0, now);
        assert!(matches!(gate.enqueue_at(now), Err(AdmissionError::Full)));
    }

    #[test]
    fn expiry_at_promotion_boundary_cannot_execute_or_double_count() {
        let gate = Admission::new(1, 2, LIMIT);
        let now = Instant::now();
        let running = gate.enqueue_at(now).unwrap();
        let mut p = claim(&running, now);
        let expired = gate.enqueue_at(now).unwrap();
        let later = gate.enqueue_at(now + Duration::from_secs(1)).unwrap();
        p.release_at(now + LIMIT);
        assert!(matches!(expired.poll_at(now + LIMIT), Err(AdmissionError::Timeout)));
        assert!(!expired.cancel_at(now + LIMIT));
        assert!(later.poll_at(now + LIMIT).unwrap().is_some());
        drop(expired);
        let s = gate.snapshot_at(now + LIMIT);
        assert_eq!((s.timed_out_total, s.cancelled_total), (1, 0));
    }

    #[test]
    fn expiry_clears_a_reservation_not_yet_claimed() {
        let gate = Admission::new(1, 1, LIMIT);
        let now = Instant::now();
        let reserved = gate.enqueue_at(now).unwrap();
        let waiting = gate.enqueue_at(now + Duration::from_secs(1)).unwrap();
        let s = gate.snapshot_at(now + LIMIT);
        assert_eq!((s.active, s.reserved, s.queued, s.timed_out_total), (0, 1, 0, 1));
        assert!(matches!(reserved.poll_at(now + LIMIT), Err(AdmissionError::Timeout)));
        let p = claim(&waiting, now + LIMIT);
        assert_eq!(p.wait_duration(), LIMIT - Duration::from_secs(1));
    }

    #[test]
    fn cancel_reserved_or_waiting_releases_space_once() {
        let gate = Admission::new(1, 2, LIMIT);
        let now = Instant::now();
        let reserved = gate.enqueue_at(now).unwrap();
        let waiting = gate.enqueue_at(now).unwrap();
        let last = gate.enqueue_at(now).unwrap();
        assert!(waiting.cancel_at(now));
        assert!(!waiting.cancel_at(now));
        assert!(reserved.cancel_at(now));
        assert!(matches!(reserved.poll_at(now), Err(AdmissionError::Cancelled)));
        assert!(last.poll_at(now).unwrap().is_some());
        drop((reserved, waiting));
        assert_eq!(gate.snapshot_at(now).cancelled_total, 2);
    }

    #[test]
    fn shrink_drains_permits_and_demotes_reservations_without_losing_fifo() {
        let gate = Admission::new(3, 1, LIMIT);
        let now = Instant::now();
        let a = gate.enqueue_at(now).unwrap();
        let b = gate.enqueue_at(now).unwrap();
        let c = gate.enqueue_at(now).unwrap();
        let d = gate.enqueue_at(now).unwrap();
        let mut pa = claim(&a, now);
        let mut pb = claim(&b, now);
        gate.set_capacity_at(1, now);
        let s = gate.snapshot_at(now);
        assert_eq!((s.active, s.reserved, s.queued), (2, 0, 2));
        assert!(c.poll_at(now).unwrap().is_none());
        assert!(matches!(gate.enqueue_at(now), Err(AdmissionError::Full)));
        pa.release_at(now);
        assert!(c.poll_at(now).unwrap().is_none(), "one live permit still fills capacity");
        pb.release_at(now);
        assert!(d.poll_at(now).unwrap().is_none());
        assert!(c.poll_at(now).unwrap().is_some());
    }

    #[test]
    fn unavailable_capacity_keeps_original_deadline_and_cannot_strand_reservations() {
        let gate = Admission::new(2, 0, LIMIT);
        let now = Instant::now();
        let a = gate.enqueue_at(now).unwrap();
        let b = gate.enqueue_at(now).unwrap();
        gate.set_capacity_at(0, now + Duration::from_secs(1));
        let s = gate.snapshot_at(now + Duration::from_secs(1));
        assert_eq!((s.active, s.reserved, s.queued), (0, 0, 2));
        assert!(a.poll_at(now + Duration::from_secs(1)).unwrap().is_none());
        gate.set_capacity_at(2, now + LIMIT);
        assert!(matches!(a.poll_at(now + LIMIT), Err(AdmissionError::Timeout)));
        assert!(matches!(b.poll_at(now + LIMIT), Err(AdmissionError::Timeout)));
        assert_eq!(gate.snapshot_at(now + LIMIT).timed_out_total, 2);
        assert!(gate.enqueue_at(now + LIMIT).is_ok());
    }

    #[test]
    fn dropped_waiter_and_permit_promote_the_next_live_request() {
        let gate = Admission::new(1, 2, LIMIT);
        let mut a = gate.enqueue().unwrap();
        let p = a.poll_wait(Duration::ZERO).unwrap().unwrap();
        let b = gate.enqueue().unwrap();
        let mut c = gate.enqueue().unwrap();
        drop(b);
        drop(p);
        let p = c.poll_wait(Duration::ZERO).unwrap().unwrap();
        assert!(matches!(c.poll_wait(Duration::ZERO), Err(AdmissionError::Cancelled)));
        drop(p);
        let s = gate.snapshot();
        assert_eq!((s.active, s.reserved, s.queued, s.cancelled_total), (0, 0, 0, 1));
    }

    #[test]
    fn concurrent_arrivals_respect_both_bounds() {
        const N: usize = 16;
        let gate = Admission::new(2, 4, LIMIT);
        let barrier = Arc::new(Barrier::new(N + 1));
        let threads: Vec<_> = (0..N).map(|_| {
            let (gate, barrier) = (gate.clone(), barrier.clone());
            std::thread::spawn(move || {
                let result = gate.enqueue();
                barrier.wait();
                result
            })
        }).collect();
        barrier.wait();
        let tickets: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        let s = gate.snapshot();
        assert_eq!((s.active, s.reserved, s.queued, s.rejected_full_total), (0, 2, 4, 10));
        assert_eq!(tickets.iter().filter(|t| t.is_ok()).count(), 6);
        drop(tickets);
        assert_eq!(gate.snapshot().cancelled_total, 6);
    }

    #[test]
    fn cancellation_racing_release_never_leaks_a_slot() {
        for _ in 0..16 {
            let gate = Admission::new(1, 1, LIMIT);
            let mut a = gate.enqueue().unwrap();
            let p = a.poll_wait(Duration::ZERO).unwrap().unwrap();
            let mut b = gate.enqueue().unwrap();
            let barrier = Arc::new(Barrier::new(2));
            let other = barrier.clone();
            let release = std::thread::spawn(move || { other.wait(); drop(p); });
            barrier.wait();
            assert!(b.cancel());
            release.join().unwrap();
            let s = gate.snapshot();
            assert_eq!((s.active, s.reserved, s.queued, s.cancelled_total), (0, 0, 0, 1));
        }
    }

    #[test]
    fn concurrent_expiry_and_release_at_deadline_count_timeout_once() {
        let gate = Admission::new(1, 1, LIMIT);
        let now = Instant::now();
        let a = gate.enqueue_at(now).unwrap();
        let mut p = claim(&a, now);
        let b = gate.enqueue_at(now).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let release = {
            let barrier = barrier.clone();
            std::thread::spawn(move || { barrier.wait(); p.release_at(now + LIMIT); })
        };
        let expiry = {
            let (barrier, gate) = (barrier.clone(), gate.clone());
            std::thread::spawn(move || { barrier.wait(); gate.snapshot_at(now + LIMIT); })
        };
        barrier.wait();
        assert!(matches!(b.poll_at(now + LIMIT), Err(AdmissionError::Timeout)));
        release.join().unwrap();
        expiry.join().unwrap();
        drop(b);
        let s = gate.snapshot_at(now + LIMIT);
        assert_eq!((s.active, s.reserved, s.queued, s.timed_out_total, s.cancelled_total), (0, 0, 0, 1, 0));
    }

    #[test]
    fn capacity_change_wakes_a_bounded_poll() {
        let gate = Admission::new(0, 1, LIMIT);
        let mut ticket = gate.enqueue().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let other = barrier.clone();
        let waiter = std::thread::spawn(move || {
            other.wait();
            ticket.poll_wait(Duration::from_secs(1)).unwrap().expect("capacity change should wake the poll")
        });
        barrier.wait();
        gate.set_capacity(1);
        let p = waiter.join().unwrap();
        assert_eq!(gate.snapshot().active, 1);
        drop(p);
        assert_eq!(gate.snapshot().active, 0);
    }
}
