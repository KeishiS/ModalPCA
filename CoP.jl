using LinearAlgebra, Statistics
include("GeometricMedian.jl");

function CoP(
  inpX::Array{Float64,2};
  α::Float64 = 0.5,
  isCentered=false,
  scale=true
)::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
  (N,d)=size(inpX);
  n = round(Int, α*N);
  X=zeros(N,d);
  gm=zeros(d);

  if !isCentered
    gm .= GM(inpX);
  end
  X .= inpX .- gm';
  if scale
    X .= Diagonal( map(i->1/norm(X[i,:]), 1:N) ) * X;
  end

  G=X*X'-Matrix(1.0I,N,N);
  ps=map(i->norm(G[:,i])^2,1:N);
  ps = ps ./ sum(ps);
  Y=X[sortperm(ps,rev=true)[1:n],:];

  eigret = (Y'*Y) ./ n |> Symmetric |> eigen;
  ds = eigret.values;
  W = eigret.vectors;
  inds = sortperm(ds, rev=true);
  loadings = W[:,inds];
  scores = X*loadings;

  return (gm, loadings, scores);
end
