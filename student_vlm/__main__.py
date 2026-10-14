import argparse
import json
from pathlib import Path
from student_vlm.experiments import config_for,evaluate_checkpoint,rank_ablation,run_all


def main():
    ap=argparse.ArgumentParser(description='Student Day 10 implementation; scripts/ and tests/ are immutable')
    ap.add_argument('command',choices=['inspect','train','evaluate','all','report'])
    ap.add_argument('--setup',choices=['E0','E1','E2','E3','E4_rank_4','E4_rank_64'],default='E1')
    ap.add_argument('--base',choices=['E1','E2','E3'],help='E4 base chosen using validation only')
    ap.add_argument('--data-root',default='data')
    ap.add_argument('--checkpoint')
    ap.add_argument('--checkpoint-root',help='Persistent Drive folder; stores latest checkpoint per experiment')
    ap.add_argument('--resume')
    ap.add_argument('--split',choices=['val','test'],default='test')
    ap.add_argument('--limit',type=int,help='Debug validation only; never abbreviated final test evidence')
    args=ap.parse_args()
    if args.command=='inspect':
        from student_vlm.inspection import inspect_data
        print(json.dumps(inspect_data(args.data_root),indent=2));return
    if args.command=='report':
        from student_vlm.reporting import prediction_figures,update_report
        update_report(args.data_root);prediction_figures(args.data_root);return
    if args.command=='all':run_all(args.data_root,args.checkpoint_root);return
    if args.setup.startswith('E4'):
        if not args.base:ap.error('--base is required for E4 (select it on validation)')
        cfg=next(c for c in rank_ablation(config_for(args.base)) if c['name']==args.setup)
    else:cfg=config_for(args.setup)
    if args.command=='train':
        if args.setup=='E0':ap.error('E0 is zero-shot, not a training experiment')
        from student_vlm.training import train
        print(train(cfg,data_root=args.data_root,resume=args.resume,checkpoint_root=args.checkpoint_root))
    else:
        if args.setup!='E0' and not args.checkpoint:ap.error('--checkpoint is required for trained setups')
        print(evaluate_checkpoint(cfg,args.checkpoint,args.split,args.data_root,args.limit))


if __name__=='__main__':main()
