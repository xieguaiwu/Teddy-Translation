import sys, os, time
BASE = '/root/smt_model'
sys.path.insert(0, BASE)
os.chdir(BASE)
import smt.data_prep as dp; dp._nlp_zh = None
import jieba; dp.tokenize_zh = lambda t: ' '.join(jieba.cut(t))
from smt.decoder import PhraseDecoder
from smt.phrase_table import load_phrase_table
from smt.language_model import KneserNeyLM

Z2E = {'beam_size':3,'stack_size':30,'max_phrase_len':4,'distortion_limit':3,'lm_weight':1.0,'translation_weight':1.0,'distortion_weight':0.3,'word_penalty':-0.5,'future_cost_estimate':False,'oov_strategy':'copy'}
E2Z = {'beam_size':3,'stack_size':30,'max_phrase_len':4,'distortion_limit':2,'lm_weight':1.0,'translation_weight':1.0,'distortion_weight':0.3,'word_penalty':-0.5,'future_cost_estimate':False,'oov_strategy':'copy'}

def clean_en(t): return ' '.join(w for w in t.split() if sum(1 for c in w if ord(c)<128)/max(len(w),1)>0.7)
def clean_zh(t):
    result=[]; prev=None
    for c in t:
        cp=ord(c); cur=(0x4E00<=cp<=0x9FFF)or(0x3000<=cp<=0x303F)or(0xFF00<=cp<=0xFFEF)or(0x30<=cp<=0x39)
        if prev is not None and cur!=prev: result.append(' ')
        result.append(c); prev=cur
    tokens=''.join(result).split()
    return ''.join(t for t in tokens if sum(1 for c in t if ord(c)>127 or c in '0123456789。，！？')/max(len(t),1)>0.5)

print('Loading ZH→EN...', flush=True)
pt_z=load_phrase_table(f'{BASE}/model/smt_zh2en_fa/phrase_table_top5.txt')
lm_z=KneserNeyLM.load(f'{BASE}/model/smt_zh2en_fa/lm.json')
d_z=PhraseDecoder(pt_z,lm_z,config=Z2E)
print(f'  {len(pt_z)} phrases', flush=True)

print('Loading EN→ZH...', flush=True)
pt_e=load_phrase_table(f'{BASE}/model/smt_en2zh_fa/phrase_table_top5.txt')
lm_e=KneserNeyLM.load(f'{BASE}/model/smt_en2zh_fa/lm.json')
d_e=PhraseDecoder(pt_e,lm_e,config=E2Z)
print(f'  {len(pt_e)} phrases', flush=True)

total,t0=0,time.time()
for group,dec,lang in [('zh_news',d_z,'zh'),('zh_lit',d_z,'zh'),('en_news',d_e,'en'),('en_lit',d_e,'en')]:
    d_in=f'{BASE}/data/source_texts/{group}'; d_out=f'{BASE}/output/smt_fa/{group}'
    os.makedirs(d_out,exist_ok=True)
    files=sorted([f for f in os.listdir(d_in) if f.endswith('.txt')])
    for fname in files:
        with open(f'{d_in}/{fname}') as f: text=f.read().strip()
        tokens=dp.tokenize(text,lang=lang).split()
        out,_=dec.decode(tokens)
        if lang=='zh': output=clean_en(' '.join(out))
        else: output=clean_zh(''.join(out))
        with open(f'{d_out}/{fname}','w') as f: f.write(output+'\n')
        total+=1
        if total%20==0: print(f'  {total}/80 ({time.time()-t0:.0f}s)', flush=True)
elapsed=time.time()-t0
print(f'DONE: {total} files in {elapsed:.0f}s ({elapsed/60:.1f}min)', flush=True)
