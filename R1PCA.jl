using LinearAlgebra, Statistics
include("GeometricMedian.jl");

function R1PCA(
  inpX::Array{Float64,2},
  maxd::Int64;
  isCentered=false
)::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
  ϵ=1e-10;
  δ=1e-10;
  Loop=100;
  function wH(x, U, c)
    norm(x-U*U'*x)<=c ? 1 : c/norm(x-U*U'*x)
  end

  (N,d)=size(inpX);
  X=zeros(N,d);
  gm=zeros(d);

  if !isCentered
    gm=GM(inpX);
  end
  X.=inpX.-gm';

  eigret = X'*X |> Symmetric |> eigen;
  λs = eigret.values;
  U₀ = eigret.vectors[:, sortperm(λs, rev=true)];
  U=zeros(d,maxd);
  U.=U₀[:,1:maxd];

  s=map(
    i->sqrt(
      max(0, dot(X[i,:], (Matrix(1.0I, d, d)-U*U')*X[i,:]))
    ),1:N
  );
  c=median(s);

  for l in 1:Loop
    q=map(i->wH(X[i,:],U, c),1:N);
    W=X'*(X.*q);
    Uˢ=qr(W*U).Q |> Matrix;
    if norm(Uˢ-U) < ϵ
      U=Uˢ;
      break;
    else
      U=Uˢ;
    end
  end

  scores = X*U;
  return (gm, U, scores);
end
