"""
1.2 Self attention from first principles on a tiny example(Numpy only)
We use T=3 tokens , d_model=4,single head
This script prints intermidiate tensors so you can trace the maths 

Dimensions summary 
X:(B=1,T=3,_model=4)
Wq/Wh/Wv:(d_model=4,d_k=2)
Q,K,V=1,3,2
Scores:1,3,3=Q@K^T
Weights:1,3,3=softmax over last dim
Outputs:1,3,2=Weights@V

"""
import numpy as np;
np.set_printoptions(precision=4,suppress=True)
# toy input 
X=np.array([[[0.1,0.2,0.3,0.4],
            [0.5,0.4,0.3,0.2],
            [0.0,0.1,0.0,0.1],
            ]],dtype=np.float32)

# Weight matrices(deduced from training in real models).Fixed for determinism
Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)
# Project to Q,K,V
Q=X@Wq
K=X@Wk
V=X@Wv

print("Qshape:",Q.shape,"\nq=\n",Q[0])
print("Kshape",K.shape,"\nk=\n",K[0])
print("Vshape",V.shape,"\nv=\n",V[0])

# Scaled dot product 
scale=1.0/np.sqrt(Q.shape[-1])
attn_scores=(Q @ K.transpose(0,2,1))*scale

# Casual Mask
mask=np.triu(np.ones((1,3,3),dtype=bool),k=1)

# Softmax over last dimension

weights=np.exp(attn_scores-attn_scores.max(axis=-1,keepdims=True))
weights=weights/weights.sum(axis=-1,keepdims=True)
print("Weights shape:",weights.shape,"Attention weights(casual)=\n",weights[0])

# Weighted Sum of V
out=weights@V
print("Output shape:",out.shape,"n\\output=\n",out[0])