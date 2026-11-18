#!/usr/bin/env python3
"""Exercise the independent Forge API with real synthetic-model execution.

No external client, pretrained quality or hardware performance is inferred.
"""
import argparse,contextlib,hashlib,http.client,importlib.util,json,os,socket,struct,subprocess,sys,tempfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
KEY="forge-native-api-validation-only"

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
fixtures=module("native_fixture_helpers",ROOT/"tests/api-coding-client.py")
scheduling=module("native_schedule_helpers",ROOT/"tests/scheduling_api.py")
checks=[]
def check(condition,message):
    if not condition:raise AssertionError(message)
def passed(message):checks.append(message)

TEMPLATE="{% for m in messages %}{{ m.role }}:{{ m.content }}{% if m.tool_calls is defined %}{{ m.tool_calls|tojson }}{% endif %}{% endfor %}assistant:"
def add_template(path):
    raw=path.read_bytes();nt,nm=struct.unpack_from('<QQ',raw,8);p=24
    def string():
        nonlocal p
        n=struct.unpack_from('<Q',raw,p)[0];p+=8;v=raw[p:p+n];p+=n;return v
    def value(t):
        nonlocal p
        if t==8:string()
        elif t==9:
            kind,count=struct.unpack_from('<IQ',raw,p);p+=12
            for _ in range(count):value(kind)
        else:p+={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
    for _ in range(nm):string();kind=struct.unpack_from('<I',raw,p)[0];p+=4;value(kind)
    metadata_end=p
    for _ in range(nt):string();nd=struct.unpack_from('<I',raw,p)[0];p+=4+8*nd+12
    payload=raw[(p+31)//32*32:]
    pack=lambda s:struct.pack('<Q',len(s.encode()))+s.encode()
    entry=pack('tokenizer.chat_template')+struct.pack('<I',8)+pack(TEMPLATE)
    header=raw[:16]+struct.pack('<Q',nm+1)+raw[24:metadata_end]+entry+raw[metadata_end:p]
    path.write_bytes(header+bytes(-len(header)%32)+payload)

@contextlib.contextmanager
def server(binary,model,parallel=2,queued=1,timeout=5,context=2048,max_tokens=256,backend=None,request_timeout=30):
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=[str(binary),'--threads','4','serve','--model',str(model),'--models-dir',str(model.parent),
         '--ctx',str(context),'--max-tokens',str(max_tokens),'--port',str(port),'--parallel',str(parallel),
         '--max-queued',str(queued),'--queue-timeout-secs',str(timeout),'--api-key',KEY,
         '--expert-ram-mib','16','--ram-margin-mib','0']+(backend or ['--cpu'])
    with tempfile.TemporaryFile() as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=log)
        def request(path,body=None,auth=True,content_type='application/json'):
            c=http.client.HTTPConnection('127.0.0.1',port,timeout=request_timeout)
            headers={'Authorization':'Bearer '+KEY} if auth else {}
            if body is not None:headers['Content-Type']=content_type
            try:
                c.request('POST' if body is not None else 'GET',path,None if body is None else json.dumps(body),headers)
                r=c.getresponse();return r.status,dict(r.getheaders()),r.read().decode()
            finally:c.close()
        try:
            deadline=time.monotonic()+60
            while True:
                check(proc.poll() is None,'native server exited during startup')
                try:
                    if request('/health')[0]==200:break
                except (OSError,http.client.HTTPException):pass
                check(time.monotonic()<deadline,'native server startup timeout');time.sleep(.025)
            yield port,request
        except BaseException:
            log.seek(0);print(log.read().decode(errors='replace'),file=sys.stderr);raise
        finally:
            proc.terminate()
            try:proc.wait(10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(10)

def body(model_id,**updates):
    value={'model':model_id,'messages':[{'role':'user','content':'hello'}],'max_tokens':4,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    value.update(updates);return value
def json_call(request,path,data=None):
    status,_,raw=request(path,data);return status,json.loads(raw)
def wait_for(request,predicate,seconds=10):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        status,data=json_call(request,'/v1/status')
        if status==200 and predicate(data):return data
        time.sleep(.01)
    raise AssertionError('expected native admission transition was not observed')

class Job:
    def __init__(self,port,data):
        self.done=threading.Event();self.raw=bytearray();self.error=None
        self.socket=socket.create_connection(('127.0.0.1',port),timeout=5);self.socket.settimeout(30)
        payload=json.dumps(data).encode();head=(f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n').encode()
        self.socket.sendall(head+payload)
        def read():
            try:
                while True:
                    x=self.socket.recv(65536)
                    if not x:break
                    self.raw.extend(x)
            except OSError as e:self.error=str(e)
            finally:self.done.set()
        self.thread=threading.Thread(target=read,daemon=True);self.thread.start()
    def cancel(self):
        try:self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:pass
        self.socket.close();self.thread.join(5)
    def status(self):
        line=bytes(self.raw).split(b'\r\n',1)[0];return int(line.split()[1]) if len(line.split())>1 else None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    binary=args.binary.resolve();start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='forge-native-api-') as td:
        directory=Path(td);plain=directory/'plain.gguf';tool=directory/'tools.gguf';literal=directory/'literal.gguf'
        source=ROOT/'tests/fixtures/tiny-qwen3moe.gguf'
        fixtures.fixture_copy(source,plain,False);fixtures.fixture_copy(source,tool,True)
        literal_text='let marker = "café 😀 </think><tool_call>example</tool_call>";'
        fixtures.TOOL_TEXT=literal_text;fixtures.fixture_copy(source,literal,True)
        for p in [plain,tool,literal]:add_template(p)
        (directory/'invalid.gguf').write_bytes(b'not a model')
        with server(binary,plain) as (port,request):
            check(request('/v1/models',auth=False)[0]==401,'API key was not enforced');passed('Bearer authentication is enforced')
            status,models=json_call(request,'/v1/models');check(status==200 and {m['id'] for m in models['data']}=={'plain.gguf','tools.gguf','literal.gguf'},str(models));passed('Catalogue exposes supported complete models and excludes malformed files')
            status,response=json_call(request,'/v1/chat/completions',body('plain.gguf'));check(status==200 and response['choices'][0]['message']['content']=='AAAA',str(response));check(response['usage']['completion_tokens']==4,str(response));passed('Native model execution returns exact nonstreaming output and usage')
            expected_executed=response['usage']['prompt_tokens']+response['usage']['completion_tokens']-1
            follow=body('plain.gguf',messages=[{'role':'user','content':'hello'},{'role':'assistant','content':'AAAA'},{'role':'user','content':'next'}])
            status,response=json_call(request,'/v1/chat/completions',follow);check(status==200 and response['usage']['prompt_tokens_details']['cached_tokens']==expected_executed,str(response));passed('Exact executed prefix is reused; the final sampled but unexecuted token is excluded')
            changed=body('plain.gguf',messages=[{'role':'user','content':'a different history'}]);status,response=json_call(request,'/v1/chat/completions',changed);check(status==200 and response['usage']['prompt_tokens_details']['cached_tokens']==0,str(response));passed('Changed conversation history resets state without reusing unrelated context')
            status,preview=json_call(request,'/api/context',body('plain.gguf'));check(status==200 and preview['prompt_tokens']>0 and preview['reserved_output_tokens']==4,str(preview));passed('Context preview uses the actual model template and tokenizer')
            for usage in [False,True]:
                status,headers,raw=request('/v1/chat/completions',body('plain.gguf',stream=True,stream_options={'include_usage':usage}))
                chunks=fixtures.events(raw);text=''.join(c['choices'][0]['delta'].get('content','') for c in chunks if c.get('choices'))
                check(status==200 and text=='AAAA' and len({c['id'] for c in chunks})==1,raw)
                check(headers.get('Content-Type','').startswith('text/event-stream') and raw.count('data: [DONE]')==1,raw)
                check([c['choices'][0]['finish_reason'] for c in chunks if c.get('choices') and c['choices'][0].get('finish_reason')]==['length'],raw)
                check(any('usage' in c for c in chunks)==usage,raw);passed(f'SSE has stable response identity, exact output and requested usage={usage}')
            status,response=json_call(request,'/v1/chat/completions',body('plain.gguf',stop=['AA']));check(status==200 and response['choices'][0]['finish_reason']=='stop' and response['choices'][0]['message']['content']=='',str(response));passed('Stop strings are withheld from the response')
            for changes in [{'model':'tools.gguf'},{'max_tokens':257},{'temperature':-1},{'tool_choice':'required'},{'response_format':{'type':'json_object'}},{'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://example.invalid/image'}}]}]}]:
                check(request('/v1/chat/completions',body('plain.gguf',**changes))[0]==400,str(changes))
            passed('Model mismatch, unsupported decoding controls, images and oversize reservations fail before inference')
            oversized=body('plain.gguf',messages=[{'role':'user','content':'z'*3000}])
            for endpoint in ['/api/context','/v1/chat/completions']:
                check(request(endpoint,oversized)[0]==400,'prompt plus output overflow accepted at '+endpoint)
            passed('Actual prompt-plus-output overflow is rejected by preview and generation before admission')
            check(request('/api/models/load',{'model':'tools.gguf'},content_type='text/plain')[0]==400,'model mutation accepted text/plain');passed('Model mutations require JSON transport')
            hidden=literal.with_suffix('.unavailable');literal.rename(hidden)
            try:
                check(request('/api/models/load',{'model':'literal.gguf'})[0]==400,'missing replacement model did not fail')
                status,response=json_call(request,'/v1/chat/completions',body('plain.gguf'))
                check(status==200 and response['choices'][0]['message']['content']=='AAAA',str(response))
                wait_for(request,lambda d:not d['admission']['model_changing'])
                passed('Failed model replacement restores the previous model and reopens admission')
            finally:hidden.rename(literal)
            status,response=json_call(request,'/api/models/load',{'model':'tools.gguf'});check(status==200,str(response));passed('Idle model switching selects a different independently loaded model')
            tool_body=body('tools.gguf',max_tokens=1,tools=[fixtures.TOOL])
            status,response=json_call(request,'/v1/chat/completions',tool_body);check(status==200,str(response));call=response['choices'][0]['message']['tool_calls'][0];check(response['choices'][0]['finish_reason']=='tool_calls' and call['function']['name']=='read_file' and json.loads(call['function']['arguments'])=={'path':'src/main.rs'},str(response));passed('Generated XML tool call becomes an OpenAI function call with an ID and JSON arguments')
            follow=body('tools.gguf',max_tokens=1,tools=[fixtures.TOOL],messages=[{'role':'user','content':'hello'},response['choices'][0]['message'],{'role':'tool','tool_call_id':call['id'],'content':'fn main() {}'}])
            check(request('/v1/chat/completions',follow)[0]==200,'tool-result continuation failed');passed('Assistant tool calls and matching tool results survive the next template turn')
            status,_,raw=request('/v1/chat/completions',dict(tool_body,stream=True));chunks=fixtures.events(raw);calls=[c['choices'][0]['delta']['tool_calls'][0] for c in chunks if c.get('choices') and 'tool_calls' in c['choices'][0]['delta']];check(status==200 and calls and calls[0]['index']==0 and calls[0]['id'],raw);passed('SSE function-call delta includes its index, ID and arguments')
            check(json_call(request,'/api/models/load',{'model':'literal.gguf'})[0]==200,'literal model load')
            literal_body=body('literal.gguf',max_tokens=1);literal_body.pop('chat_template_kwargs')
            status,response=json_call(request,'/v1/chat/completions',literal_body);check(status==200 and response['choices'][0]['message']['content']==literal_text,str(response));passed('Literal reasoning/tool tags in code remain text when no matching protocol is active')
            status,_,raw=request('/v1/chat/completions',dict(literal_body,stream=True,max_tokens=3))
            chunks=fixtures.events(raw);streamed=''.join(c['choices'][0]['delta'].get('content','') for c in chunks if c.get('choices'))
            check(status==200 and streamed==literal_text*3,raw);passed('Incremental SSE preserves Unicode and literal code across generated tokens')
            status,response=json_call(request,'/v1/chat/completions',dict(literal_body,max_tokens=3,stop=[';let']))
            check(status==200 and response['choices'][0]['message']['content']==literal_text[:-1] and response['usage']['completion_tokens']==2,str(response))
            passed('A stop crossing generated-token boundaries is withheld without truncating prior Unicode')
            status,stats=json_call(request,'/v1/status');check(status==200 and stats['admission']['active']==0 and stats['admission']['queued']==0,str(stats));passed('Completed and rejected requests leave no admission permits behind')
        expanded_dir=directory/'queue';expanded_dir.mkdir();expanded=expanded_dir/'expanded.gguf';scheduling.expand_fixture(source,expanded);add_template(expanded)
        for capacity in [1,4]:
            with server(binary,expanded,parallel=capacity,queued=1,timeout=2,context=32768,max_tokens=16384) as (port,request):
                jobs=[]
                try:
                    for n in range(capacity):
                        jobs.append(Job(port,body('expanded.gguf',max_tokens=16384,stream=True,messages=[{'role':'user','content':f'long request {n}'}])))
                        wait_for(request,lambda d:d['admission']['active']==n+1)
                    cancelled=Job(port,body('expanded.gguf',max_tokens=4,stream=True));jobs.append(cancelled)
                    wait_for(request,lambda d:d['admission']['queued']==1)
                    cancelled.cancel()
                    wait_for(request,lambda d:d['admission']['queued']==0 and d['admission']['active']==capacity)
                    passed(f'Capacity {capacity} removes a disconnected queued request before admission')
                    waiting=Job(port,body('expanded.gguf',max_tokens=4,stream=True));jobs.append(waiting)
                    wait_for(request,lambda d:d['admission']['queued']==1)
                    check(request('/v1/chat/completions',body('expanded.gguf'))[0]==429,'full queue not rejected')
                    check(request('/api/models/load',{'model':'expanded.gguf'})[0]==409,'busy model switch accepted')
                    jobs[0].cancel();check(waiting.done.wait(10),'queued request did not make progress after cancellation');check(waiting.status()==200,bytes(waiting.raw).decode(errors='replace'))
                    passed(f'Capacity{capacity} executes real independent requests, queues overflow and admits waiting work after cancellation')
                finally:
                    for job in jobs:job.cancel()
                wait_for(request,lambda d:d['admission']['active']==0 and d['admission']['queued']==0)
                passed(f'Capacity{capacity} cancellation releases active and queued resources')
        with server(binary,expanded,parallel=1,queued=1,timeout=1,context=32768,max_tokens=16384) as (port,request):
            active=Job(port,body('expanded.gguf',max_tokens=16384,stream=True));wait_for(request,lambda d:d['admission']['active']==1)
            try:
                start_wait=time.monotonic();status,_,raw=request('/v1/chat/completions',body('expanded.gguf'));check(status==504 and time.monotonic()-start_wait<5,raw);passed('Waiting request deadline returns504 and removes its queue entry')
            finally:active.cancel()
            wait_for(request,lambda d:d['admission']['active']==0 and d['admission']['queued']==0)
    project_version=next(line.split('"')[1] for line in (ROOT/'Cargo.toml').read_text().splitlines() if line.startswith('version = "'))
    engine_version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    report={'version':project_version,'engine_version':engine_version,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'status':'passed','checks_passed':len(checks),'checks':checks,'elapsed_seconds':time.monotonic()-start,
        'backend':'independent CPU runtime','gpu_tested':False,'real_model_quality_tested':False,'external_coding_client_tested':False,
        'scope':'Actual HTTP/SSE/model execution, sequence reuse, model switching and1/4active admission with cancellation/timeout; synthetic fixtures only'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
