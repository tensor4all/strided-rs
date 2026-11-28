import json,glob,sys
d=sys.argv[1]
print("cell dtype T atoms | ratio | S ms (total) | L ms | vjp parts x/w/y ms")
for f in sorted(glob.glob(d+'/cg-*.jsonl')):
    r=json.loads(open(f).read().splitlines()[-1]); t=r['timings']
    L=t['llmlip']['median_ns']
    if 'strided_total_ns' in t: S=t['strided_total_ns']; p="/".join(f"{t[k]['median_ns']/1e6:.1f}" for k in('strided_vjp_x','strided_vjp_w','strided_vjp_y'))
    else: S=t['strided']['median_ns']; p=""
    print(f"{r['cell']} {r['dtype']} {r['threads']}T {r['atoms']} | {S/L:.3f} | {S/1e6:.2f} | {L/1e6:.2f} | {p}")
