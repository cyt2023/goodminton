"""Local LLM selects evidence-linked review priorities; app renders numeric facts."""
import argparse
import json
from pathlib import Path
from analysis_jobs import atomic_json
from vision_model import MODEL,MODEL_ID,MODEL_REVISION

FOCUS={
 'review_wrist_path':'沿叠加轨迹回看手腕的变化，先判断是否存在遮挡或跟踪跳点。',
 'review_elbow_projection':'回看触球附近的肘部投影姿态；这项二维角度不能直接判断三维动作是否正确。',
 'review_foot_path':'结合完整动作窗口回看双脚路径，并与同一机位的同类动作比较。'
}


def generate_feedback(request):
    event=request['event'];observations=event['quality_metrics']['observations_2d']
    allowed={k:v for k,v in observations.items() if v is not None}
    if not allowed:raise ValueError('没有足够的可见测量，保留原有证据不足提示')
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model,processor=load(str(MODEL),trust_remote_code=False,local_files_only=True)
    prompt='''你协助用户回看羽毛球视频。输入只有真实二维测量，没有三维标定、已验证参考动作或质量评分。
不要评判好坏、发力、拍面、受伤风险，也不要生成新数字。只选择最多三个回看重点，按相关性排序。
仅输出JSON：{"items":[{"evidence_id":"输入中的一个键","focus_code":"review_wrist_path或review_elbow_projection或review_foot_path"}]}。
wrist开头或包含wrist的键仅对应review_wrist_path；elbow仅对应review_elbow_projection；foot仅对应review_foot_path。
测量如下：'''+json.dumps(allowed,ensure_ascii=False)
    formatted=apply_chat_template(processor,model.config,prompt=prompt,num_images=0)
    result=generate(model,processor,formatted,max_tokens=200,temperature=0,verbose=False)
    text=result.text;start=text.find('{')
    if start<0:raise ValueError('本地模型未返回结构化回看重点')
    parsed,_=json.JSONDecoder().raw_decode(text[start:]);items=[]
    for entry in parsed.get('items',[])[:3]:
        key,focus=entry.get('evidence_id'),entry.get('focus_code')
        expected='review_wrist_path' if key and 'wrist' in key else 'review_elbow_projection' if key and 'elbow' in key else 'review_foot_path' if key and 'foot' in key else None
        if key not in allowed or focus!=expected or any(v['evidence_id']==key for v in items):continue
        items.append({'evidence_id':key,'measured_value':allowed[key],'focus_code':focus,'text':FOCUS[focus]})
    if not items:raise ValueError('模型未选出与真实测量对应的回看重点')
    return {'status':'completed','evidence_hash':request['evidence_hash'],'event_id':event['event_id'],
        'engine':MODEL_ID,'model_revision':MODEL_REVISION,'llm_generated':True,'generation_scope':'LLM selects review priorities; measured facts and explanatory text are rendered from validated IDs',
        'items':items,'text':' '.join(i['text'] for i in items),'quality_assessment_available':False,
        'evidence':event['coaching_feedback']['evidence'],'raw_response':text}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();request=json.loads(Path(a.input).read_text())
    atomic_json(Path(request['output']),generate_feedback(request))
