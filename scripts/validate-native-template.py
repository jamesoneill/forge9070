#!/usr/bin/env python3
"""Compare the native renderer to Python Jinja using a supplied official template.

Requires Jinja2 for this development-only check. It downloads no model weights.
The output records the template hash and results, not the third-party template.
"""
import argparse, hashlib, json, os, re, subprocess, tempfile, time
from pathlib import Path
import jinja2

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cargo', default='cargo')
    a = p.parse_args()
    source = a.template.read_text()
    transformed = re.sub(r'{%(-?)\s*(?:endgeneration|generation)\s*(-?)%}',
        lambda m: '{%' + m[1] + ' set __forge_generation_annotation = true ' + m[2] + '%}', source)
    env = jinja2.Environment(autoescape=False, extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=False)
    def fail(s): raise ValueError(s)
    env.globals['raise_exception'] = fail
    template = env.from_string(transformed)
    tool = {'type':'function','function':{'name':'read_file','description':'Read UTF-8 <source>',
        'parameters':{'type':'object','properties':{'path':{'type':'string'},
        'range':{'type':'array','items':{'type':'integer'}}},'required':['path']}}}
    cases = []
    def add(name, messages, thinking=True, tools=None):
        request = {'model':'template-validation','max_tokens':4,'messages':messages,
                   'chat_template_kwargs':{'enable_thinking':thinking}}
        if tools: request['tools'] = tools
        # Compare the same canonical JSON maps used by the native request parser.
        request = json.loads(json.dumps(request, sort_keys=True))
        normalized = json.loads(json.dumps(request['messages']))
        for m in normalized:
            if m.get('content') is None: m['content'] = ''
            for call in m.get('tool_calls', []):
                if isinstance(call['function']['arguments'], str):
                    call['function']['arguments'] = json.loads(call['function']['arguments'])
            if m['role'] == 'tool': m['name'] = 'read_file'
        values = dict(messages=normalized, add_generation_prompt=True, enable_thinking=thinking,
                      preserve_thinking=True, add_vision_id=False, bos_token='', eos_token='<|im_end|>')
        if tools: values['tools'] = request['tools']
        cases.append(dict(name=name, request=request, expected=template.render(**values)))
    add('plain_nonreasoning', [{'role':'user','content':'Explain café <tags>.'}], False)
    add('plain_reasoning', [{'role':'user','content':'Review this Rust code.'}])
    add('system_and_history', [{'role':'system','content':' Be precise. '},
        {'role':'user','content':'first'},{'role':'assistant','content':'answer','reasoning_content':'prior plan'},
        {'role':'user','content':'next'}])
    add('tools_initial', [{'role':'user','content':'Read src/main.rs'}], False, [tool])
    add('tool_result', [{'role':'user','content':'Read file'},
        {'role':'assistant','content':None,'reasoning_content':'inspect it','tool_calls':[{'id':'c1','type':'function',
        'function':{'name':'read_file','arguments':json.dumps({'path':'src/café.rs','range':[1,9]},sort_keys=True)}}]},
        {'role':'tool','tool_call_id':'c1','content':'fn main() { /* <ok> */ }'}], True, [tool])
    add('parallel_tool_results', [{'role':'user','content':'Compare both'},
        {'role':'assistant','content':'','tool_calls':[{'id':f'c{n}','type':'function',
        'function':{'name':'read_file','arguments':'{"path":"file'+str(n)+'"}'}} for n in [1,2]]},
        {'role':'tool','tool_call_id':'c1','content':'one'},{'role':'tool','tool_call_id':'c2','content':'two'}], False, [tool])
    started = time.monotonic()
    command = [a.cargo,'test','-p','forge-engine','--lib','protocol::tests::external_template_byte_parity',
               '--locked','--offline','--','--ignored','--nocapture']
    with tempfile.TemporaryDirectory(prefix='forge-template-parity-') as td:
        reference = Path(td)/'cases.json'
        reference.write_text(json.dumps({'template':source,'cases':cases}, ensure_ascii=False))
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ,FORGE_TEMPLATE_PARITY_JSON=str(reference)),
                                capture_output=True, text=True, timeout=300)
    report = {'status':'passed' if result.returncode == 0 else 'failed',
        'template_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'reference':'Python Jinja2 '+jinja2.__version__+' with Transformers default tojson',
        'canonical_json_map_order':True,'cases':[c['name'] for c in cases],
        'elapsed_seconds':time.monotonic()-started,'command':command,
        'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'scope':'Prompt render byte parity; text and tool histories only, no inference or image support'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(result.returncode)

if __name__ == '__main__': main()
