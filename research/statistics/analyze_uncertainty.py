"""Conditional cluster bootstrap of frozen predictions; no fitting or API calls.
Inputs are prepared by build_milestone.py. Clusters are sampled with replacement
without outcome stratification. All models in a dataset share bootstrap draws.
Intervals do not account for training/split selection variability or establish
participant identity. Endpoints remain recording/image weighted.
"""
import argparse,csv,json,itertools
from pathlib import Path
import numpy as np

def analyze(rows,reps,seed):
 groups=sorted({r['group'] for r in rows}); models=sorted({r['model'] for r in rows})
 # group x model x [tn,fp,fn,tp,unclassified_healthy,unclassified_pd]
 counts=np.zeros((len(groups),len(models),6),dtype=int)
 gi={g:i for i,g in enumerate(groups)};mi={m:i for i,m in enumerate(models)}
 for r in rows:
  y=int(r['actual']);pred=r['predicted'];k=4+y if pred=='' else 2*y+int(pred)
  counts[gi[r['group']],mi[r['model']],k]+=1
 # Identical observation set per model; no silent row dropping.
 ids={m:{r['observation_id'] for r in rows if r['model']==m} for m in models}
 assert all(v==ids[models[0]] for v in ids.values())
 rng=np.random.default_rng(seed);weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=reps)
 def metrics(c):
  tn,fp,fn,tp,uh,up=np.moveaxis(c,-1,0);covered=tn+fp+fn+tp;total=covered+uh+up
  def div(a,b):return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b!=0)
  sen=div(tp,tp+fn);spe=div(tn,tn+fp)
  return dict(accuracy=div(tn+tp,covered),sensitivity=sen,specificity=spe,balanced_accuracy=(sen+spe)/2,coverage=div(covered,total),correct_among_all=div(tn+tp,total))
 point=metrics(counts.sum(0));boot=metrics(np.einsum('bg,gmc->bmc',weights,counts))
 out=[];pairs=[]
 for j,m in enumerate(models):
  for metric,values in boot.items():
   v=values[:,j];finite=v[np.isfinite(v)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
   out.append(dict(model=m,metric=metric,estimate=point[metric][j],ci_lower=lo,ci_upper=hi,valid_replicates=len(finite),replicates=reps,groups=len(groups),observations=len(ids[m])))
 for a,b in itertools.combinations(range(len(models)),2):
  for metric in ['accuracy','sensitivity','specificity','coverage','correct_among_all']:
   dif=boot[metric][:,a]-boot[metric][:,b];finite=dif[np.isfinite(dif)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
   pairs.append(dict(model_a=models[a],model_b=models[b],metric=metric,difference_a_minus_b=point[metric][a]-point[metric][b],ci_lower=lo,ci_upper=hi,valid_replicates=len(finite),replicates=reps))
 return out,pairs

def writecsv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([{k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in r.items()} for r in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path(__file__).parent/'inputs');p.add_argument('--output',type=Path,default=Path(__file__).parent/'results');p.add_argument('--replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=20261029);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 allrows=[];allpairs=[]
 for path in sorted(a.inputs.glob('*.csv')):
  with path.open() as f: rows=list(csv.DictReader(f))
  o,q=analyze(rows,a.replicates,a.seed)
  allrows.extend(dict(dataset=path.stem,**r) for r in o);allpairs.extend(dict(dataset=path.stem,**r) for r in q)
 writecsv(a.output/'confidence_intervals.csv',allrows);writecsv(a.output/'paired_differences.csv',allpairs)
 (a.output/'bootstrap_protocol.json').write_text(json.dumps(dict(seed=a.seed,replicates=a.replicates,confidence=.95,method='unstratified cluster percentile bootstrap of fixed predictions; common draws for paired models',conditioning='saved scores, folds and fitted predictions held fixed; no retraining',estimand='recording/image-weighted pooled metrics, not patient-level classification',limits=['Grouping identities not independently verified','Not a full uncertainty analysis of CV training or model selection','Kaggle uses image-level resampling; participants unknown; previously inspected test','No multiplicity-adjusted superiority tests','Undefined accuracy under zero coverage remains blank']),indent=2))
 print('Wrote',len(allrows),'metric intervals and',len(allpairs),'paired differences')
if __name__=='__main__':main()
