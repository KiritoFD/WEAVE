import pathlib

p = pathlib.Path("sbflow/train.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    """def estimate_moments(A,B,eps,batch,ot_ratio,ot_reg,n_batches,gen,ot_cost):
    mom=MomentAccumulator()
    for _ in range(n_batches):
        x,y,_=pick_pair(A,B,batch,ot_ratio,ot_reg,gen,ot_cost)""",
    """def estimate_moments(A,B,eps,batch,ot_ratio,ot_reg,n_batches,gen,ot_cost,ot_iters=200,ot_tol=0.0):
    mom=MomentAccumulator()
    for _ in range(n_batches):
        x,y,_=pick_pair(A,B,batch,ot_ratio,ot_reg,gen,ot_cost,ot_iters,ot_tol)""",
)

s = s.replace(
    """def loss_batch(model,A,B,mom,epsilon,batch,ot_ratio,ot_reg,min_t,gen,loss_mode,ot_cost,want_info=True):
    z0,z1,info=pick_pair(A,B,batch,ot_ratio,ot_reg,gen,ot_cost)""",
    """def loss_batch(model,A,B,mom,epsilon,batch,ot_ratio,ot_reg,min_t,gen,loss_mode,ot_cost,want_info=True,
               ot_iters=200,ot_tol=0.0):
    z0,z1,info=pick_pair(A,B,batch,ot_ratio,ot_reg,gen,ot_cost,ot_iters,ot_tol)""",
)

s = s.replace(
    """    moments=estimate_moments(A,B,args.eps,args.microbatch,args.ot_ratio,args.ot_reg,
                            args.moments_batches,gen,args.ot_cost)""",
    """    moments=estimate_moments(A,B,args.eps,args.microbatch,args.ot_ratio,args.ot_reg,
                            args.moments_batches,gen,args.ot_cost,args.ot_iters,args.ot_tol)""",
)

s = s.replace(
    """                                     args.ot_ratio,args.ot_reg,args.min_t,gen,args.loss_mode,args.ot_cost,
                                     want_info=want_info)""",
    """                                     args.ot_ratio,args.ot_reg,args.min_t,gen,args.loss_mode,args.ot_cost,
                                     want_info=want_info,ot_iters=args.ot_iters,ot_tol=args.ot_tol)""",
)

p.write_text(s, encoding="utf-8")
print("patched train.py")
