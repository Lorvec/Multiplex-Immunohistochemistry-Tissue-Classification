"""MIHIC reproducibility pipeline for compact CNN classification experiments.

The module implements the dataset audit, compact architectures, knowledge distillation,
training, calibration, efficiency benchmarking, final test evaluation, and report generation
used by the accompanying reproducibility workflow.
"""
from __future__ import annotations
import argparse, collections, contextlib, csv, hashlib, json, math, os, platform, random, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

CLASSES = ['Immune cells','Necrosis','Other','Stroma','Tumor','alveoli','background']
EXPECTED = {'train':[6630,14468,37288,25023,91357,7636,12599],
            'val':[1817,1857,11188,5362,24625,1262,3149],
            'test':[2393,7694,12937,7592,27993,2891,3937]}
STAINS = ['Cyclin-D1','D2-40','CD20','CD34','CD38','CD68','CDK4','Ki67','CD3','FAP','P53','SMA']
MEAN = (.485,.456,.406); STD = (.229,.224,.225)
SIZES = {'nano':((24,40,80,160),(1,2,3,1),3.,.15),
         'tiny':((24,48,96,192),(1,2,3,2),3.,.20),
         'small':((32,64,128,256),(2,3,4,2),4.,.25)}
FAMILY_NAMES = [f'{sz}_{"single" if scale=="single" else "multi"}_{kd}'
                for sz in ('nano','tiny','small')
                for scale in ('single','multi')
                for kd in ('noKD','KD')]
COMPARATORS = ['teacher','repvit','mobilenetv4','shvit','efficientnetb0','mobilenetv3','tinycnn']
FULL_REGISTRY = COMPARATORS + FAMILY_NAMES
PILOT_REGISTRY = ['teacher','nano_single_noKD','nano_single_KD',
                  'tiny_single_noKD','tiny_single_KD','small_single_noKD','small_single_KD',
                  'repvit','mobilenetv4']
N_CLASSES = len(CLASSES)


def jwrite(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(tmp,path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def find_root(raw):
    root=Path(raw).expanduser().resolve()
    if all((root / n).is_dir() for n in ('train','val','test')): return root
    if root.is_dir():
        found=[p.parent for p in root.rglob('train') if (p.parent/'val').is_dir() and (p.parent/'test').is_dir()]
        if len(found)==1: return found[0]
    raise FileNotFoundError(f'Expected train/val/test under {root}')


def analyze_stain(path):
    name=Path(path).name.upper()
    for stain in STAINS:
        if f'-{stain.upper()}-' in name: return stain
    return 'unknown'


def hint_source(filename):
    # For exploratory overlap indication ONLY, not patient/region identity.
    stem=Path(filename).stem
    return re.sub(r'-\d+$','',stem)


def audit(root, output, strict=True):
    output.mkdir(parents=True,exist_ok=True)
    records=[]; counts=[]; splits={}
    for split in ('train','val','test'):
        base=root/split
        names=sorted(p.name for p in base.iterdir() if p.is_dir())
        if names != CLASSES: raise RuntimeError(f'Class mismatch: {split}: {names} != {CLASSES}')
        for klass,n in zip(CLASSES,EXPECTED[split]):
            pp=sorted(p for p in (base/klass).rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'})
            counts.append({'split':split,'class_name':klass,'found':len(pp),'expected':n})
            if strict and len(pp)!=n: raise RuntimeError(f'Count mismatch: {split}/{klass} {len(pp)} != {n}')
            for p in pp:
                records.append({'split':split,'class_name':klass,'sample_id':p.relative_to(root).as_posix(),
                                'basename':p.name,'source_hint':hint_source(p.name),
                                'stain':analyze_stain(p.name),'bytes':p.stat().st_size})
    frame=pd.DataFrame(records); pd.DataFrame(counts).to_csv(output/'class_counts.csv',index=False)
    frame.to_csv(output/'manifest.csv.gz',index=False,compression='gzip')
    bysplit={s: set(frame.loc[frame.split==s,'source_hint']) for s in ('train','val','test')}
    pairs={'train_val':len(bysplit['train']&bysplit['val']),
           'train_test':len(bysplit['train']&bysplit['test']),
           'val_test':len(bysplit['val']&bysplit['test'])}
    digest=hashlib.sha256('\n'.join(f'{z.sample_id}|{z.bytes}' for z in frame.itertuples()).encode()).hexdigest()
    info={'root':str(root),'count':len(frame),'counts':frame.groupby('split').size().to_dict(),
          'manifest_sha256':digest,'source_hint_overlap':pairs,'source_hint_not_verified_identity':True,
          'patient_disjointness_verified':False,'test_images_opened':False,'no_augmentation':True}
    jwrite(output/'audit.json',info)
    print(pd.crosstab(frame.class_name,frame.split).to_string())
    print('\nAUDIT:\n'+json.dumps(info,indent=2))
    return info


class IDFolder(datasets.ImageFolder):
    def __getitem__(self,index):
        img,label=super().__getitem__(index)
        return img,label,index


def create_datasets(root, device, workers):
    transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train=IDFolder(root/'train',transform=transform)
    val=IDFolder(root/'val',transform=transform)
    if train.classes!=CLASSES or val.classes!=CLASSES: raise RuntimeError('Unexpected class mapping')
    kw=dict(num_workers=workers,pin_memory=device.type=='cuda',persistent_workers=workers>0)
    return train,val,kw


class ConvBNAct(nn.Sequential):
    def __init__(self,ci,co,k=1,stride=1,groups=1,act=True):
        layers=[nn.Conv2d(ci,co,k,stride,k//2,groups=groups,bias=False),nn.BatchNorm2d(co)]
        if act: layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SE(nn.Module):
    def __init__(self,c):
        super().__init__(); h=max(8,c//4)
        self.avg=nn.AdaptiveAvgPool2d(1); self.reduce=nn.Conv2d(c,h,1); self.expand=nn.Conv2d(h,c,1)
    def forward(self,x): return x*torch.sigmoid(self.expand(F.silu(self.reduce(self.avg(x)))))


def divisible(v): return max(8,int(v+4)//8*8)


class UIB(nn.Module):
    def __init__(self,ci,co,stride=1,expand=3.,multi=True):
        super().__init__(); h=divisible(ci*expand); self.residual=(stride==1 and ci==co)
        self.multi=multi; self.expand=ConvBNAct(ci,h)
        if multi:
            self.n1=h//2
            self.dw3=ConvBNAct(self.n1,self.n1,3,stride,self.n1)
            self.dw5=ConvBNAct(h-self.n1,h-self.n1,5,stride,h-self.n1)
        else: self.dw=ConvBNAct(h,h,3,stride,h)
        self.se=SE(h); self.project=ConvBNAct(h,co,1,act=False)
    def forward(self,x):
        z=self.expand(x)
        if self.multi:
            a,b=torch.split(z,[self.n1,z.shape[1]-self.n1],1)
            z=torch.cat([self.dw3(a),self.dw5(b)],1)
        else: z=self.dw(z)
        z=self.project(self.se(z))
        return z+x if self.residual else z


class CompactCNN(nn.Module):
    """Same no-attention Nano/Tiny/Small family and stage dimensions as archived MIHIC model."""
    def __init__(self,size,scale):
        super().__init__(); widths,depths,exp,drop=SIZES[size]
        self.stem=ConvBNAct(3,widths[0],3,2)
        cin=widths[0]; stages=[]
        for co,d in zip(widths,depths):
            blocks=[]
            for idx in range(d):
                blocks.append(UIB(cin,co,stride=2 if idx==0 else 1,expand=exp,multi=(scale=='multi')))
                cin=co
            stages.append(nn.Sequential(*blocks))
        self.stages=nn.ModuleList(stages)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(nn.Dropout(drop),nn.Linear(cin,N_CLASSES))
    def forward(self,x):
        x=self.stem(x)
        for stage in self.stages: x=stage(x)
        return self.classifier(self.pool(x).flatten(1))


class TinyCNNControl(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; ci=3
        for i,co in enumerate([32,64,128,256]):
            layers.extend([nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True)])
            if i<3: layers.append(nn.MaxPool2d(2))
            ci=co
        self.features=nn.Sequential(*layers); self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(256,N_CLASSES)
    def forward(self,x): return self.classifier(self.pool(self.features(x)).flatten(1))


def make_model(name, pretrained=True):
    """Return (model, pretrained). Pretrained weights may download once on the user's computer."""
    if name=='teacher':
        m=convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2]=nn.Linear(m.classifier[2].in_features,N_CLASSES)
        return m,True
    if name=='tinycnn': return TinyCNNControl(),False
    if name in COMPARATORS:
        if name in ('efficientnetb0','mobilenetv3'):
            from torchvision.models import (efficientnet_b0, EfficientNet_B0_Weights,
                                            mobilenet_v3_small, MobileNet_V3_Small_Weights)
            if name=='efficientnetb0':
                m=efficientnet_b0(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,N_CLASSES)
            else:
                m=mobilenet_v3_small(weights=None); m.classifier[3]=nn.Linear(m.classifier[3].in_features,N_CLASSES)
            return m,False
        import timm
        names={'repvit':'repvit_m0_9.dist_450e_in1k',
               'mobilenetv4':'mobilenetv4_conv_small.e1200_r224_in1k',
               'shvit':'shvit_s2.in1k'}
        try: m=timm.create_model(names[name],pretrained=pretrained,num_classes=N_CLASSES)
        except Exception as e:
            raise RuntimeError(f'Could not load model {name} ({names[name]}): {e}') from e
        return m,True
    m=re.fullmatch(r'(nano|tiny|small)_(single|multi)_(KD|noKD)',name)
    if m: return CompactCNN(m.group(1),m.group(2)),False
    raise KeyError(name)


def logits_of(output):
    if torch.is_tensor(output): return output
    if isinstance(output,(tuple,list)):
        candidates=[t for t in output if torch.is_tensor(t)]
        if candidates: return torch.stack(candidates).mean(0)
    if isinstance(output,dict):
        for key in ('logits','out','prediction'):
            if key in output and torch.is_tensor(output[key]): return output[key]
    raise TypeError(f'Unexpected logits: {type(output)}')


def dataloader(ds,batch,workers_kwargs,shuffle=False,seed=42):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),**workers_kwargs)


def accuracy_calibration(y,p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-10,1.);p=p/p.sum(1,keepdims=True)
    pred=p.argmax(1);correct=(pred==y).astype(np.float64)
    conf=p.max(1); edges=np.linspace(0,1,16)
    ece=0.
    for i in range(15):
        mask=(conf>=edges[i]) & ((conf<edges[i+1]) if i<14 else (conf<=edges[i+1]))
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    onehot=np.eye(N_CLASSES)[y]; brier=np.mean(np.sum((p-onehot)**2,axis=1))
    return {'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'mcc':float(matthews_corrcoef(y,pred)),'kappa':float(cohen_kappa_score(y,pred)),
            'nll':float(log_loss(y,p,labels=list(range(N_CLASSES)))),
            'ece15':float(ece),'brier':float(brier),
            'auroc':float(roc_auc_score(y,p,labels=list(range(N_CLASSES)),multi_class='ovr',average='macro')) if len(np.unique(y))==N_CLASSES else float('nan')}


@torch.no_grad()
def infer(model,loader,device,amp=False):
    model.eval(); all_z=[];ys=[];ids=[]
    for x,y,i in loader:
        x=x.to(device,non_blocking=True)
        with torch.autocast('cuda',enabled=amp and device.type=='cuda'):
            z=logits_of(model(x))
        all_z.append(z.float().cpu().numpy());ys.append(y.numpy());ids.append(i.numpy())
    return np.concatenate(ids),np.concatenate(ys),np.concatenate(all_z)


def softmax_np(logits, temp=1.):
    z=np.asarray(logits,dtype=np.float64)/temp; z=z-z.max(1,keepdims=True)
    z=np.exp(z); return z/z.sum(1,keepdims=True)


def fit_temperature(y,logits):
    from scipy.optimize import minimize_scalar
    def nll(logtemp):
        return log_loss(y,softmax_np(logits,float(np.exp(logtemp))),labels=list(range(N_CLASSES)))
    result=minimize_scalar(nll,bounds=(math.log(.2),math.log(10.)),method='bounded',options={'xatol':1e-5})
    valtemp=float(np.exp(result.x))
    return valtemp if result.success and nll(math.log(valtemp)) < nll(0.) else 1.


def sample_predictions(dataset,indices,y,logits,root):
    paths=[Path(dataset.samples[int(i)][0]) for i in indices]
    p=softmax_np(logits)
    data={'sample_id':[v.relative_to(root).as_posix() for v in paths],
          'target':y,'prediction':p.argmax(1),'stain':[analyze_stain(v) for v in paths]}
    for c in range(N_CLASSES):
        data[f'logit_{c}']=logits[:,c];data[f'prob_{c}']=p[:,c]
    return pd.DataFrame(data)


def per_group_metrics(table,group_key):
    records=[]
    for group,g in table.groupby(group_key,sort=True):
        y=g.target.to_numpy(); pred=g.prediction.to_numpy()
        ps,rs,fs,sp=precision_recall_fscore_support(y,pred,labels=list(range(N_CLASSES)),zero_division=0)
        if group_key=='stain':
            present=np.unique(np.concatenate((y,pred)))
            macro=float(f1_score(y,pred,labels=present,average='macro',zero_division=0))
        else:macro=float(f1_score(y,pred,average='macro',zero_division=0))
        records.append({group_key:group,'n':len(g),'macro_f1':macro,
                        'accuracy':float(accuracy_score(y,pred)),
                        'present_classes':int(len(np.unique(y)))})
    return pd.DataFrame(records)


def save_prediction_reports(df,out):
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    y=df.target.to_numpy();yp=df.prediction.to_numpy()
    precision,recall,fs,support=precision_recall_fscore_support(y,yp,labels=list(range(N_CLASSES)),zero_division=0)
    pd.DataFrame({'class_name':CLASSES,'precision':precision,'recall':recall,'f1':fs,'support':support}).to_csv(out/'per_class.csv',index=False)
    cm=confusion_matrix(y,yp,labels=list(range(N_CLASSES)))
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_counts.csv')
    pd.DataFrame(cm/np.maximum(cm.sum(1,keepdims=True),1),index=CLASSES,columns=CLASSES).to_csv(out/'confusion_normalized.csv')
    per_group_metrics(df,'stain').to_csv(out/'per_stain.csv',index=False)
    # One-vs-rest classwise ECE: explicitly separate from top-label ECE.
    p=df[[f'prob_{i}' for i in range(N_CLASSES)]].to_numpy()
    records=[]
    edges=np.linspace(0,1,16)
    for k,klass in enumerate(CLASSES):
        binary=(y==k).astype(float); confidences=p[:,k]
        ece=0.
        for i in range(15):
            mask=(confidences>=edges[i])&((confidences<edges[i+1]) if i<14 else (confidences<=edges[i+1]))
            if mask.any():ece+=mask.mean()*abs(binary[mask].mean()-confidences[mask].mean())
        records.append({'class_name':klass,'classwise_ece15_ovr':float(ece),'support':int(binary.sum())})
    pd.DataFrame(records).to_csv(out/'per_class_calibration.csv',index=False)


def sha256file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def teacher_cache(root,out,seed,train,worker_kw,device,eval_batch):
    cachefile=out/'teacher_logits'/f'seed_{seed}.npz'
    ckpt=out/'runs'/'teacher'/f'seed_{seed}'/'best.pt'
    if not ckpt.exists(): raise RuntimeError(f'Missing teacher checkpoint {ckpt}')
    manifest=json.loads((out/'audit'/'audit.json').read_text())['manifest_sha256']
    ckhash=sha256file(ckpt)
    if cachefile.exists():
        with np.load(cachefile) as d:
            if (d['logits'].shape==(len(train),N_CLASSES) and
                str(d['manifest'].item())==manifest and str(d['teacher_sha'].item())==ckhash):
                return torch.from_numpy(d['logits'].copy())
        raise RuntimeError('Stale teacher-logits cache; remove only this cache AFTER verifying dataset/checkpoint')
    net,_=make_model('teacher',pretrained=False)
    net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['state_dict'])
    net.to(device)
    loader=dataloader(train,eval_batch,worker_kw,shuffle=False)
    ids,y,logits=infer(net,loader,device,amp=device.type=='cuda')
    assert np.array_equal(ids,np.arange(len(train))), 'Teacher logit cache index order mismatch'
    cachefile.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cachefile,logits=logits.astype(np.float32),manifest=manifest,teacher_sha=ckhash)
    del net
    if device.type=='cuda':torch.cuda.empty_cache()
    print('Saved aligned no-augmentation teacher targets:',cachefile)
    return torch.from_numpy(logits.astype(np.float32))


def save_checkpoint(path,model,name,seed,epoch,config_hash):
    torch.save({'state_dict':model.state_dict(),'model':name,'seed':seed,'epoch':epoch,
                'classes':CLASSES,'no_augmentation':True,'config_sha256':config_hash},path)


def train_one(name,seed,root,out,args,train,val,worker_kw,device,cache=None):
    place=out/'runs'/name/f'seed_{seed}'
    place.mkdir(parents=True,exist_ok=True)
    done=place/'summary.json'
    if done.exists():
        old=json.loads(done.read_text())
        if old['config_sha256']!=args.config_sha256:raise RuntimeError(f'Config changed for {done}, use a new output root')
        print('RESUME SKIP:',name,seed)
        return old
    seed_all(seed)
    model,is_pretrained=make_model(name,pretrained=True)
    model.to(device)
    use_kd=name.endswith('_KD')
    if use_kd and cache is None:raise RuntimeError(f'KD cache missing for {name}, seed {seed}')
    if use_kd and cache.shape!=(len(train),N_CLASSES):raise RuntimeError('KD cache shape incorrect')
    lr=args.lr_pretrained if is_pretrained else args.lr_scratch
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    counts=np.bincount(np.asarray(train.targets),minlength=N_CLASSES)
    weights=torch.from_numpy((counts**(-.5)).astype(np.float32)).to(device)
    weights=weights/weights.mean()
    tloader=dataloader(train,args.batch,worker_kw,shuffle=True,seed=seed)
    vloader=dataloader(val,args.eval_batch,worker_kw,shuffle=False)
    best_f1=-1.; best_ep=0;bad=0;history=[];start_epoch=1;started=time.monotonic()
    last=place/'last.pt'
    if last.exists():
        old=torch.load(last,map_location='cpu',weights_only=False)
        if old['config_sha256']!=args.config_sha256:raise RuntimeError('Incompatible interrupted run config')
        model.load_state_dict(old['state_dict']);optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        best_f1=float(old['best_f1']);best_ep=int(old['best_ep']);bad=int(old['bad'])
        history=old['history'];start_epoch=int(old['epoch'])+1
        torch.set_rng_state(old['torch_rng']);np.random.set_state(old['np_rng']);random.setstate(old['py_rng'])
        tloader.generator.set_state(old['loader_rng'])
        if device.type=='cuda' and old.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(old['cuda_rng'])
        print('RESUMING interrupted run at epoch',start_epoch)
    try:
        for epoch in range(start_epoch,args.epochs+1):
            if bad>=args.patience:break
            scale=(epoch/max(1,args.warmup) if epoch<=args.warmup else .5*(1+math.cos(math.pi*(epoch-args.warmup)/max(1,args.epochs-args.warmup))))
            for group in optimizer.param_groups:group['lr']=lr*scale
            model.train(); loss_total=0.;n=0;all_preds=[];all_targets=[]
            for x,y,idx in tloader:
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',enabled=device.type=='cuda'):
                    z=logits_of(model(x))
                    ce=F.cross_entropy(z,y,weight=weights,label_smoothing=args.label_smoothing)
                    if use_kd:
                        t=cache[idx].to(device,non_blocking=True)
                        T=args.temperature
                        kdloss=F.kl_div(F.log_softmax(z.float()/T,dim=1),F.softmax(t.float()/T,dim=1),reduction='batchmean')*(T*T)
                        loss=(1-args.alpha)*ce+args.alpha*kdloss
                    else:loss=ce
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                n+=len(y);loss_total+=loss.detach().float().item()*len(y)
                all_preds.append(z.detach().argmax(1).cpu().numpy());all_targets.append(y.cpu().numpy())
            ids,v_y,v_z=infer(model,vloader,device,amp=device.type=='cuda')
            valstats=accuracy_calibration(v_y,softmax_np(v_z))
            train_f1=float(f1_score(np.concatenate(all_targets),np.concatenate(all_preds),average='macro',zero_division=0))
            rec={'epoch':epoch,'train_loss':loss_total/max(n,1),'train_f1_online':train_f1,**valstats}
            history.append(rec)
            if valstats['macro_f1']>best_f1+1e-7:
                best_f1=valstats['macro_f1'];best_ep=epoch;bad=0
                save_checkpoint(place/'best.pt',model,name,seed,epoch,args.config_sha256)
            else:bad+=1
            # Exact resume at epoch boundary; torch_rng and loader RNG saved.
            torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                        'best_f1':best_f1,'best_ep':best_ep,'bad':bad,'epoch':epoch,'history':history,
                        'torch_rng':torch.get_rng_state(),'np_rng':np.random.get_state(),
                        'py_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                        'loader_rng':tloader.generator.get_state(),'config_sha256':args.config_sha256},last)
            pd.DataFrame(history).to_csv(place/'history.csv',index=False)
            print(f'{name:25} seed={seed} epoch={epoch:02d} trainF1={train_f1:.4f} valF1={best_f1:.4f} valNLL={valstats["nll"]:.4f}',flush=True)
            if bad>=args.patience:break
        if not (place/'best.pt').exists():raise RuntimeError('Training completed without best checkpoint')
        model.load_state_dict(torch.load(place/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        ids,y,z=infer(model,vloader,device,amp=device.type=='cuda')
        raw=accuracy_calibration(y,softmax_np(z))
        temp=fit_temperature(y,z)
        calibrated=accuracy_calibration(y,softmax_np(z,temp))
        report=sample_predictions(val,ids,y,z,root)
        save_prediction_reports(report,place/'validation')
        jwrite(place/'calibration.json',{'temperature_val_fitted':temp,'raw_val_metrics':raw,
                                       'calibrated_val_metrics_optimistic_do_not_report_as_independent':calibrated})
        elapsed=time.monotonic()-started
        meta={'model':name,'seed':seed,'phase':args.phase,'uses_kd':use_kd,
              'pretrained':is_pretrained,'parameters':sum(p.numel() for p in model.parameters()),
              'train_seconds_this_execution':elapsed,'best_epoch':best_ep,
              'epochs_run':len(history),'no_augmentation':True,'config_sha256':args.config_sha256,
              'train_f1_online_last':history[-1]['train_f1_online'],'train_validation_gap_online_approx':history[-1]['train_f1_online']-raw['macro_f1'],
              'checkpoint_MB':(place/'best.pt').stat().st_size/1e6,'temperature_val_fitted':temp,**raw}
        jwrite(done,meta)
        # Retain last.pt for reproducible resumption record; disk may be large.
        return meta
    finally:
        del model
        if device.type=='cuda':torch.cuda.empty_cache()


def registry(phase):return PILOT_REGISTRY if phase=='pilot' else FULL_REGISTRY


def build_protocol(args,root,out):
    protocol={'phase':args.phase,'seed_list':args.seeds,'max_epochs':args.epochs,'patience':args.patience,
              'batch_size':args.batch,'eval_batch_size':args.eval_batch,'num_workers':args.workers,
              'lr_pretrained':args.lr_pretrained,'lr_scratch':args.lr_scratch,'label_smoothing':args.label_smoothing,
              'weight_decay':args.weight_decay,'warmup':args.warmup,'kd_alpha':args.alpha,'kd_T':args.temperature,
              'architecture':'original widths/depths, attention off; 3x3 single or 3x3 + 5x5 split-channel multi',
              'data_root':str(root),'no_data_augmentation':True,
              'transform':'Resize((128,128)), ToTensor, ImageNet normalization only',
              'prespecified_reg':registry(args.phase),
              'test_not_used_for_model_selection':True,
              'patient_independence':'unverified'}
    encoded=json.dumps(protocol,sort_keys=True).encode()
    args.config_sha256=hashlib.sha256(encoded).hexdigest()
    protocol['config_sha256']=args.config_sha256
    if (out/'protocol.json').exists():
        old=json.loads((out/'protocol.json').read_text())
        if old['config_sha256']!=args.config_sha256:raise RuntimeError('Output folder belongs to another protocol! Choose NEW output root')
    else:jwrite(out/'protocol.json',protocol)
    return protocol


def run_train(args,root,out):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not args.allow_cpu:raise RuntimeError('CUDA GPU required for full training (optional --allow-cpu for small smoke datasets).')
    seed_all(42)
    audit(root,out/'audit',strict=not args.allow_count_mismatch)
    protocol=build_protocol(args,root,out)
    jwrite(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
              'timm':getattr(__import__('timm'),'__version__','unknown'),'platform':platform.platform(),
              'cpu':platform.processor(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'cuda':torch.version.cuda})
    train,val,worker_kw=create_datasets(root,device,args.workers)
    order=registry(args.phase)
    for seed in args.seeds:
        for name in order:
            print('\n'+'='*70+'\n'+f'Running {name}, seed {seed}, phase {args.phase}',flush=True)
            cache=teacher_cache(root,out,seed,train,worker_kw,device,args.eval_batch) if name.endswith('_KD') else None
            train_one(name,seed,root,out,args,train,val,worker_kw,device,cache)
            gather_summary(out)
    print('\nTRAINING FINISHED; test pixels never loaded. See validation_summary.csv and plots.')


def gather_summary(out):
    records=[]
    for p in sorted((out/'runs').glob('*/seed_*/summary.json')):records.append(json.loads(p.read_text()))
    if records:
        df=pd.DataFrame(records)
        df.sort_values(['model','seed']).to_csv(out/'validation_by_seed.csv',index=False)
        fields=['macro_f1','accuracy','balanced_accuracy','auroc','ece15','nll','brier','parameters','checkpoint_MB']
        summary=df.groupby('model')[fields].agg(['mean','std'])
        summary.columns=['_'.join(t) for t in summary.columns]
        summary['n_seeds']=df.groupby('model').size()
        summary.sort_values('macro_f1_mean',ascending=False).to_csv(out/'validation_summary.csv')
        print(summary[['macro_f1_mean','macro_f1_std','parameters_mean','n_seeds']].round(4).to_string())


def eval_test(args,root,out):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['phase']!='full':raise RuntimeError('Test evaluation is allowed only for completed FULL phase')
    expected={(model,seed) for model in FULL_REGISTRY for seed in protocol['seed_list']}
    completed={(p.parent.parent.name,int(p.parent.name.split('_')[1])) for p in (out/'runs').glob('*/seed_*/summary.json')}
    missing=expected-completed
    if missing:raise RuntimeError(f'Missing {len(missing)} full runs: {sorted(missing)[:6]}')
    if (out/'TEST_EVALUATION_COMPLETED.json').exists():
        print('Test already evaluated in this output root. Skipping to prevent re-evaluation.');return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not args.allow_cpu:raise RuntimeError('CUDA recommended for test evaluation.')
    seed_all(42)
    transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ds=IDFolder(root/'test',transform=transform)
    if ds.classes!=CLASSES:raise RuntimeError('Test class ordering mismatch')
    kw=dict(num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    loader=dataloader(ds,args.eval_batch,kw)
    res=[]
    for name,seed in sorted(expected):
        net,_=make_model(name,pretrained=False)
        ckpt=out/'runs'/name/f'seed_{seed}'/'best.pt'
        net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['state_dict'])
        net.to(device)
        ids,y,raw_logits=infer(net,loader,device,amp=device.type=='cuda')
        report=sample_predictions(ds,ids,y,raw_logits,root)
        target=out/'test_results'/name/f'seed_{seed}'
        save_prediction_reports(report,target)
        raw=accuracy_calibration(y,softmax_np(raw_logits))
        temperature=json.loads((out/'runs'/name/f'seed_{seed}'/'calibration.json').read_text())['temperature_val_fitted']
        scaled=accuracy_calibration(y,softmax_np(raw_logits,temperature))
        result={'model':name,'seed':seed,'temperature_val_fitted':temperature,
                **raw,**{f'temperature_scaled_{k}':v for k,v in scaled.items()}}
        res.append(result); jwrite(target/'test_metrics.json',result)
        print('TEST',name,seed,'F1',round(raw['macro_f1'],5),'NLL',round(raw['nll'],5))
        del net
        if device.type=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(res).to_csv(out/'test_by_seed.csv',index=False)
    jwrite(out/'TEST_EVALUATION_COMPLETED.json',{'datetime_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'evaluated_models':len(res),'note':'Final test evaluation performed after validation-based checkpoint selection.'})


def gmaccounter(model,device):
    # Counts conv/linear MACs; will not claim full-model operator coverage.
    total=[0]
    hooks=[]
    def conv_hook(m,args,y):
        total[0]+=y.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
    def linear_hook(m,args,y):total[0]+=y.numel()*m.in_features
    for module in model.modules():
        if isinstance(module,nn.Conv2d):hooks.append(module.register_forward_hook(conv_hook))
        if isinstance(module,nn.Linear):hooks.append(module.register_forward_hook(linear_hook))
    model.eval()
    try:
        with torch.no_grad():model(torch.randn(1,3,128,128,device=device))
    finally:
        for h in hooks:h.remove()
    return total[0]/1e9


def benchmark_one(name,seed,out,args,device):
    meta=json.loads((out/'runs'/name/f'seed_{seed}'/'summary.json').read_text())
    model,_=make_model(name,pretrained=False)
    model.load_state_dict(torch.load(out/'runs'/name/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    model.to(device);model.eval()
    params=sum(x.numel() for x in model.parameters())
    try: gmac=gmaccounter(model,device)
    except Exception as e: warnings.warn(f'GMAC estimate failed for {name}: {e}');gmac=float('nan')
    def bench(batch,loops,warmup):
        x=torch.randn(batch,3,128,128,device=device)
        if device.type=='cuda':torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(warmup):model(x)
            if device.type=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            for _ in range(loops):model(x)
            if device.type=='cuda':torch.cuda.synchronize()
            dt=time.perf_counter()-t
        return dt/loops*1000
    gpu_b1=bench(1,100,30) if device.type=='cuda' else float('nan')
    gpu_b64=bench(64,60,20) if device.type=='cuda' else float('nan')
    gpu_throughput=64*1000/gpu_b64 if device.type=='cuda' else float('nan')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats();bench(64,8,2);peak_MB=torch.cuda.max_memory_allocated()/1e6
    else:peak_MB=float('nan')
    model.to('cpu');device_cpu=torch.device('cpu');device=device_cpu
    old_threads=torch.get_num_threads();torch.set_num_threads(8)
    try: cpu_b1=bench(1,40,10)
    finally: torch.set_num_threads(old_threads)
    return {'model':name,'seed':seed,'parameters':params,
            'approx_conv_linear_GMACs':gmac,'GPU_batch1_ms':gpu_b1,'GPU_batch64_img_per_s':gpu_throughput,
            'GPU_peak_inference_MB':peak_MB,'CPU_batch1_ms':cpu_b1,
            'checkpoint_MB':meta['checkpoint_MB'],
            'estimate_note':'GMACs conv+linear layer hooks only; synthetic FP32 model-only input (not end-to-end).'}


def run_bench(args,out):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not (out/'protocol.json').exists():raise RuntimeError('Missing protocol')
    plan=json.loads((out/'protocol.json').read_text())
    seed=plan['seed_list'][0]
    done=sorted((out/'runs').glob(f'*/seed_{seed}/summary.json'))
    rows=[]
    old=out/'efficiency.csv'
    if old.exists():rows=pd.read_csv(old).to_dict('records')
    seen={(r['model'],int(r['seed'])) for r in rows}
    for summary in done:
        model=summary.parent.parent.name
        if (model,seed) in seen:continue
        row=benchmark_one(model,seed,out,args,device)
        rows.append(row);pd.DataFrame(rows).to_csv(old,index=False)
        print('BENCH',model,row['CPU_batch1_ms'],'ms CPU batch1')
    print('Hardware-specific model-only benchmark saved:',old)


def plot_confusion(df,title,path):
    import matplotlib.pyplot as plt
    c=confusion_matrix(df.target,df.prediction,labels=list(range(N_CLASSES)))
    c=c/np.maximum(c.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(c,vmin=0,vmax=1)
    ax.set_xticks(range(N_CLASSES),CLASSES,rotation=45,ha='right')
    ax.set_yticks(range(N_CLASSES),CLASSES)
    ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(title)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):ax.text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def reliability_plot(df,title,path,temperature=1.):
    import matplotlib.pyplot as plt
    z=df[[f'logit_{i}' for i in range(N_CLASSES)]].to_numpy()
    p=softmax_np(z,temperature);confidence=p.max(1);correct=(p.argmax(1)==df.target.to_numpy()).astype(int)
    xs=[];acc=[];n=[]
    for left,right in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>=left)&(confidence<(right if right<1 else 1.000001))
        if mask.any():xs.append(confidence[mask].mean());acc.append(correct[mask].mean());n.append(mask.sum())
    fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],'--',label='Perfect calibration')
    ax.scatter(xs,acc,s=np.maximum(10,np.array(n)/max(n)*100),label='Observed')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Fraction correct',title=title)
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def paired_differences(folder,split='validation'):
    if split=='validation':pred_root=folder/'runs'
    else:pred_root=folder/'test_results'
    compares=[]
    for size in ('nano','tiny','small'):
        for scale in ('single','multi'):
            compares.append((f'{size}_{scale}_KD',f'{size}_{scale}_noKD','KD minus noKD'))
        for kd in ('KD','noKD'):
            compares.append((f'{size}_multi_{kd}',f'{size}_single_{kd}','multiscale minus single'))
    for model in ('small_single_KD','small_multi_KD','nano_multi_KD','repvit','mobilenetv4'):
        compares.append((model,'teacher','candidate minus teacher'))
    rows=[]
    for a,b,purpose in compares:
        for seed in [42,123,2026]:
            def read(name):
                f=(folder/'runs'/name/f'seed_{seed}'/'validation'/'predictions.csv.gz' if split=='validation'
                   else folder/'test_results'/name/f'seed_{seed}'/'predictions.csv.gz')
                return pd.read_csv(f) if f.exists() else None
            df1,df2=read(a),read(b)
            if df1 is None or df2 is None:continue
            x=df1.set_index('sample_id').sort_index();y=df2.set_index('sample_id').sort_index()
            if not x.index.equals(y.index) or not (x.target.values==y.target.values).all():
                raise RuntimeError(f'Paired prediction sample mismatch: {a} vs {b}, seed {seed}')
            m1=accuracy_calibration(x.target.values,x[[f'prob_{k}' for k in range(N_CLASSES)]].to_numpy())
            m2=accuracy_calibration(y.target.values,y[[f'prob_{k}' for k in range(N_CLASSES)]].to_numpy())
            rows.append({'split':split,'a':a,'b':b,'seed':seed,'comparison':purpose,
                         **{f'delta_{metric}':m1[metric]-m2[metric] for metric in ('macro_f1','accuracy','auroc','nll','ece15','brier')}})
    return pd.DataFrame(rows)


def cluster_bootstrap(one,two,clusters,reps=400,random_seed=31415):
    # Explicit VERIFIED patient/region mapping required; no patch-iid uncertainty.
    assert one.index.equals(two.index)
    rng=np.random.default_rng(random_seed)
    groups={g:np.where(clusters==g)[0] for g in np.unique(clusters)}
    keys=np.array(list(groups.keys()));diffs=[]
    for _ in range(reps):
        pick=rng.choice(keys,size=len(keys),replace=True)
        indices=np.concatenate([groups[x] for x in pick])
        yp=one.iloc[indices].target.to_numpy()
        a=one.iloc[indices].prediction.to_numpy();b=two.iloc[indices].prediction.to_numpy()
        diffs.append(f1_score(yp,a,average='macro',labels=list(range(N_CLASSES)),zero_division=0)-
                     f1_score(yp,b,average='macro',labels=list(range(N_CLASSES)),zero_division=0))
    return np.percentile(diffs,[2.5,97.5])


def report(out,cluster_map=None,bootstrap_reps=400):
    import matplotlib.pyplot as plt
    figs=out/'paper_figures';tables=out/'paper_tables';figs.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    summary=out/'validation_summary.csv'
    if not summary.exists():raise RuntimeError('Run the complete training workflow before reporting.')
    v=pd.read_csv(summary)
    v.to_csv(tables/'validation_metrics_mean_std.csv',index=False)
    for phase in ('validation','test'):
        d=paired_differences(out,phase)
        if not d.empty:
            d.to_csv(tables/f'{phase}_paired_seed_differences.csv',index=False)
            ds=d.groupby(['a','b','comparison']).agg({k:['mean','std'] for k in d.columns if k.startswith('delta_')})
            ds.columns=['_'.join(x) for x in ds.columns]
            ds.reset_index().to_csv(tables/f'{phase}_paired_summary.csv',index=False)
    if (out/'test_by_seed.csv').exists():
        raw=pd.read_csv(out/'test_by_seed.csv')
        raw.to_csv(tables/'test_model_seed_results.csv',index=False)
        keys=['macro_f1','accuracy','balanced_accuracy','auroc','nll','ece15','brier',
              'temperature_scaled_nll','temperature_scaled_ece15']
        agg=raw.groupby('model')[keys].agg(['mean','std'])
        agg.columns=['_'.join(k) for k in agg.columns]
        agg.reset_index().to_csv(tables/'test_metrics_mean_sd.csv',index=False)
    if (out/'efficiency.csv').exists():
        pd.read_csv(out/'efficiency.csv').to_csv(tables/'hardware_model_only_efficiency.csv',index=False)
    for phase in ('validation','test'):
        folder=(out/'runs' if phase=='validation' else out/'test_results')
        for model in ('teacher','repvit','small_single_KD','small_multi_KD','nano_multi_KD','mobilenetv4'):
            f=(folder/model/'seed_42'/'validation'/'predictions.csv.gz' if phase=='validation'
               else folder/model/'seed_42'/'predictions.csv.gz')
            if f.exists():
                df=pd.read_csv(f)
                plot_confusion(df,f'{phase}: {model}',figs/f'{phase}_{model}_confusion.png')
                reliability_plot(df,f'{phase}: {model} raw',figs/f'{phase}_{model}_reliability.png')
                if phase=='test':
                    m=json.loads((out/'runs'/model/'seed_42'/'calibration.json').read_text())
                    reliability_plot(df,f'{phase}: {model} scaled',figs/f'{phase}_{model}_reliability_scaled.png',m['temperature_val_fitted'])
    # Scatter tradeoff + train curves
    fig,ax=plt.subplots(figsize=(7,5))
    p=v.dropna(subset=['parameters_mean','macro_f1_mean'])
    ax.scatter(p.parameters_mean/1e6,p.macro_f1_mean)
    for row in p.itertuples():ax.annotate(row.model,(row.parameters_mean/1e6,row.macro_f1_mean),fontsize=6)
    ax.set(xlabel='Trainable parameters (millions)',ylabel='Validation Macro-F1',title='Model size vs performance (validation)')
    ax.set_xscale('log');fig.tight_layout();fig.savefig(figs/'validation_parameter_performance.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for model in ['teacher','nano_single_KD','nano_single_noKD','small_single_KD','small_single_noKD']:
        f=out/'runs'/model/'seed_42'/'history.csv'
        if f.exists():
            hist=pd.read_csv(f);ax.plot(hist.epoch,hist.macro_f1,label=model)
    ax.set(xlabel='Epoch',ylabel='Validation Macro-F1',title='Training histories (seed 42)')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figs/'validation_learning_curves.png',dpi=220);plt.close(fig)
    # Fully enumerated layer-level architecture specification for reproducibility.
    layers=[];registry_rows=[]
    for size in ('nano','tiny','small'):
        for scale in ('single','multi'):
            model=CompactCNN(size,scale)
            modelid=f'{size}_{scale}'
            registry_rows.append({'model':modelid,'parameters':sum(p.numel() for p in model.parameters()),
                                  'depthwise_scale':scale,'attention':False,'input_pixels':128})
            for module_path,module in model.named_modules():
                if not module_path:continue
                owned=sum(p.numel() for p in module.parameters(recurse=False))
                if isinstance(module,nn.Conv2d):
                    attrs={'in_channels':module.in_channels,'out_channels':module.out_channels,
                           'kernel':str(module.kernel_size),'stride':str(module.stride),
                           'groups':module.groups}
                elif isinstance(module,nn.Linear):
                    attrs={'in_channels':module.in_features,'out_channels':module.out_features,
                           'kernel':'','stride':'','groups':''}
                else:attrs={'in_channels':'','out_channels':'','kernel':'','stride':'','groups':''}
                layers.append({'model':modelid,'layer':module_path,'layer_type':type(module).__name__,
                               'owned_parameters':owned,**attrs})
    pd.DataFrame(layers).to_csv(tables/'complete_layer_level_specs.csv',index=False)
    pd.DataFrame(registry_rows).to_csv(tables/'family_parameter_counts.csv',index=False)
    # Export complete per-class and per-stain seed-level tables for supplementary material.
    for phase in ('validation','test'):
        base=out/'runs' if phase=='validation' else out/'test_results'
        rows_class=[];rows_stain=[];rows_cal=[]
        for pred_file in sorted(base.glob('*/seed_*/validation/per_class.csv') if phase=='validation' else base.glob('*/seed_*/per_class.csv')):
            model=pred_file.parents[2].name if phase=='validation' else pred_file.parents[1].name
            seed=int(pred_file.parent.parent.name.split('_')[1]) if phase=='validation' else int(pred_file.parent.name.split('_')[1])
            cl=pd.read_csv(pred_file);cl.insert(0,'seed',seed);cl.insert(0,'model',model)
            rows_class.append(cl)
            stain_file=pred_file.with_name('per_stain.csv')
            if stain_file.exists():
                st=pd.read_csv(stain_file);st.insert(0,'seed',seed);st.insert(0,'model',model);rows_stain.append(st)
            cal_file=pred_file.with_name('per_class_calibration.csv')
            if cal_file.exists():
                ca=pd.read_csv(cal_file);ca.insert(0,'seed',seed);ca.insert(0,'model',model);rows_cal.append(ca)
        if rows_class:pd.concat(rows_class,ignore_index=True).to_csv(tables/f'{phase}_per_class_by_seed.csv',index=False)
        if rows_stain:pd.concat(rows_stain,ignore_index=True).to_csv(tables/f'{phase}_per_stain_by_seed.csv',index=False)
        if rows_cal:pd.concat(rows_cal,ignore_index=True).to_csv(tables/f'{phase}_per_class_calibration.csv',index=False)
        if rows_class:
            dat=pd.concat(rows_class,ignore_index=True)
            chosen=[n for n in ('teacher','repvit','mobilenetv4','nano_single_KD','tiny_single_KD','small_single_KD','nano_multi_KD','tiny_multi_KD','small_multi_KD') if n in dat.model.unique()]
            if chosen:
                fig,ax=plt.subplots(figsize=(10,5))
                im=dat[dat.model.isin(chosen)].pivot_table(index='model',columns='class_name',values='f1',aggfunc='mean').reindex(index=chosen,columns=CLASSES)
                obj=ax.imshow(im,vmin=0,vmax=1)
                ax.set_xticks(range(N_CLASSES),CLASSES,rotation=45,ha='right')
                ax.set_yticks(range(len(chosen)),chosen);ax.set_title(f'{phase} mean per-class F1')
                for i in range(len(chosen)):
                    for j in range(N_CLASSES):
                        value=im.iloc[i,j]
                        if np.isfinite(value):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=7)
                fig.colorbar(obj,ax=ax);fig.tight_layout();fig.savefig(figs/f'{phase}_per_class_F1_heatmap.png',dpi=220);plt.close(fig)
        if rows_stain:
            dat=pd.concat(rows_stain,ignore_index=True)
            chosen=[n for n in ('teacher','repvit','small_single_KD') if n in dat.model.unique()]
            if chosen:
                im=dat[dat.model.isin(chosen)].pivot_table(index='model',columns='stain',values='macro_f1',aggfunc='mean').reindex(index=chosen)
                fig,ax=plt.subplots(figsize=(10,3.8));obj=ax.imshow(im,aspect='auto',vmin=0,vmax=1)
                ax.set_xticks(range(len(im.columns)),im.columns,rotation=40,ha='right')
                ax.set_yticks(range(len(im)),im.index);ax.set_title(f'{phase} stain Macro-F1: descriptive; class supports differ')
                fig.colorbar(obj,ax=ax);fig.tight_layout();fig.savefig(figs/f'{phase}_stain_descriptive_heatmap.png',dpi=220);plt.close(fig)
        # Compare KD, architecture factor over three seeds without assuming patient-level inference.
        raw=out/f'{phase}_by_seed.csv'
        if raw.exists():
            data=pd.read_csv(raw)
            for measure in ('macro_f1','nll','ece15'):
                filtered=data[data.model.str.match(r'^(nano|tiny|small)_(single|multi)_(noKD|KD)$')]
                if not filtered.empty:
                    grouped=filtered.groupby('model')[measure].agg(['mean','std'])
                    mm=grouped['mean'].sort_values()
                    err=grouped['std'].reindex(mm.index).fillna(0)
                    fig,ax=plt.subplots(figsize=(9,5))
                    ax.barh(range(len(mm)),mm.values,xerr=err.values)
                    ax.set_yticks(range(len(mm)),mm.index,fontsize=8);ax.set_xlabel(measure)
                    ax.set_title(f'{phase} {measure}: mean ± seed SD')
                    fig.tight_layout();fig.savefig(figs/f'{phase}_architectures_{measure}.png',dpi=220);plt.close(fig)
    if (out/'efficiency.csv').exists():
        dat=pd.read_csv(out/'efficiency.csv')
        if len(dat):
            perf=v.set_index('model')
            data=dat[dat.model.isin(perf.index)].copy()
            data['validation_f1']=data.model.map(perf['macro_f1_mean'])
            fig,ax=plt.subplots(figsize=(8,5))
            ax.scatter(data.approx_conv_linear_GMACs,data.validation_f1)
            for r in data.itertuples():ax.annotate(r.model,(r.approx_conv_linear_GMACs,r.validation_f1),fontsize=6)
            ax.set(xlabel='Approx. conv+linear GMACs (not full-model)',ylabel='Validation Macro-F1',title='Approximate compute-performance trade-off')
            fig.tight_layout();fig.savefig(figs/'validation_approx_GMAC_tradeoff.png',dpi=220);plt.close(fig)
    # Cluster-aware uncertainty is ALWAYS opt-in with confirmed IDs.
    if cluster_map is not None:
        mapping=pd.read_csv(cluster_map)
        if not {'sample_id','cluster_id'}<=set(mapping.columns):raise RuntimeError('Cluster file requires sample_id,cluster_id')
        if mapping.cluster_id.isna().any() or mapping.cluster_id.nunique()<2:raise RuntimeError('Invalid independent cluster IDs')
        ci=[]
        for phase in ('validation','test'):
            for model in ('small_multi_KD','nano_multi_KD','small_single_KD'):
                for seed in (42,123,2026):
                    def load(n):
                        f=(out/'runs'/n/f'seed_{seed}'/'validation'/'predictions.csv.gz' if phase=='validation'
                           else out/'test_results'/n/f'seed_{seed}'/'predictions.csv.gz')
                        if not f.exists():return None
                        return pd.read_csv(f).set_index('sample_id').sort_index()
                    a=load(model);b=load('teacher')
                    if a is None or b is None:continue
                    if not a.index.equals(b.index):raise RuntimeError('Mismatch in cluster bootstrap sample IDs')
                    idx=a.index.to_frame(index=False).merge(mapping,on='sample_id',how='left')
                    if idx.cluster_id.isna().any():
                        raise RuntimeError('Missing verified cluster ID for some samples; cannot compute CI')
                    lo,hi=cluster_bootstrap(a,b,idx.cluster_id.to_numpy(),bootstrap_reps)
                    ci.append({'phase':phase,'model':model,'reference':'teacher','seed':seed,
                               'n_independent_clusters':idx.cluster_id.nunique(),'ci_low':lo,'ci_high':hi})
        if ci:pd.DataFrame(ci).to_csv(tables/'verified_cluster_bootstrap_CI.csv',index=False)
    jwrite(out/'report_notes.json',{'metrics_note':'3-seed standard deviations are NOT patient-level uncertainty estimates.',
            'calibration_note':'Temperature fitted on validation, calibration gain on validation is optimistically biased; test after freeze is reported.',
            'patient_generalization':'Unverified in official MIHIC region-grouped data.',
            'cluster_ci':str(cluster_map) if cluster_map else 'NOT COMPUTED: requires independently verified patient/region cluster map.',
            'figures_and_tables_created':True})
    print('PAPER FILES READY:\n',figs,'\n',tables)


def smoke(out):
    # Tiny synthetic tests for model shapes and early math; no MIHIC data.
    dev=torch.device('cpu');seed_all(123)
    for size in ('nano','tiny','small'):
        for scale in ('single','multi'):
            net=CompactCNN(size,scale)
            net.eval()
            with torch.no_grad():z=net(torch.randn(2,3,128,128))
            assert z.shape==(2,N_CLASSES)
            print(size,scale,'params',sum(v.numel() for v in net.parameters()))
    y=np.array([0,1,2,3,4,5,6]);logits=np.eye(N_CLASSES)*3
    p=softmax_np(logits)
    assert accuracy_calibration(y,p)['accuracy']==1.
    test=np.array([0,1,2,3,4,5,6])
    t=fit_temperature(test,logits)
    print('SMOKE passed: architectures, probability metrics, temperature optimizer; T=',t)


