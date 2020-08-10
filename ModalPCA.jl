using LinearAlgebra, Statistics, StatsBase, LineSearches, RCall
R"library(robustbase)"

"""
    ModalPCA(X::Array{Float64,2}; Nc::Int64=5, Ng::Int64=20, seed::Int64=100)

ModalPCA is the proposed method in our paper.

`X` in argments denotes data, `\\mathbb{R}^{N\times d}`. `N` and `d` represents the number of observations and dimensions, respectively. `Nc` and `Ng` is parameters used for GRID algorithm.

The return values consist of *center vector(`\bm{c}`)* , *principal components(PC)* and *scores* . The principal components represents `\\left\\{ \\hat{\\bm{v}}_k \\right\\}_{k=1}^d` in our paper. The center vector and scores are calculated by:
```math
\\begin{aligned}
  \\bm{c} = \\sum_{k=1}^{d} \\hat{m}_k \\hat{\bm{v}}_k
,\\\\
  scores = \\left( X - \\bm{1} \\bm{c}^{\\top} \\right) \\text{PC}
\\end{aligned}
.
```
"""
function ModalPCA(
  X::Array{Float64,2};
  Nc::Int64 = 5, Ng::Int64 = 20, seed::Int64 = 100
)::Tuple{Array{Float64,1}, Array{Float64,2}, Array{Float64,2}}
  (N,d) = size(X);
  W = zeros(d,d);

  function ϕ(z)::Float64
    exp(-z*z*0.5) / sqrt(2π)
  end
  function ϕₕ(z,h)::Float64
    ϕ(z/h)/h
  end
  function Fₙ(v::Vector; data::Array{Float64,2}=X)::Float64
    y=data*v;
    h = 1.144 * mad(y) * N^(-0.2);
    if h < 1e-10
      1e+30
    else
      m = uniMode(y);
      sum(ϕₕ.(y .- m, h)) / N
    end
  end
  function logFₙ(v::Vector; data::Array{Float64,2}=X)::Float64
    Fₙ(v, data=data) |> log
  end
  function Fₙ(
    v::Vector,w::Vector;
    data::Array{Float64,2}=X
  )::Float64
    y = data*v;
    h = 1.144*mad(y)*N^(-0.2);
    if h < 1e-10
      1e+30
    else
      m = uniMode(y);
      sum(ϕₕ.(y .- m, h) .* (w./sum(w)))
    end
  end
  function logFₙ(
    v::Vector, w::Vector;
    data::Array{Float64,2}=X
  )::Float64
    Fₙ(v, w, data=data) |> log
  end

  function searchcomp(
    v₀::Vector;
    subW = nothing, data::Array{Float64,2} = X, ϵ::Float64=1e-8, Loop::Int64=100
  )::Tuple{Float64, Array{Float64,1}}

    local (N, d) = size(data);
    U = if subW == nothing
      nullspace(reshape(v₀, 1, length(v₀)))
      # nullspace(v₀')
    else
      nullspace(hcat(v₀, subW)')
    end
    
    function P₀φ₀(v)
      (U'*v) ./ (1 + dot(v₀, v))
    end
    function φ₀⁻¹P₀⁻¹(β)
      ( (U*β).*2.0 + v₀.*(1-dot(β,β)) ) ./ (1 + dot(β, β))
    end
    function H(v)
      dot(v, data'*(data.*qⁿ)*v) - 2.0 * mⁿ * dot(v, data'*qⁿ)
    end
    function ∂φ₀⁻¹P₀⁻¹∂β(β)
      (( I.*(1 + dot(β, β)) - β*β'.*2.0 )*U') .* (2/(1+dot(β, β))^2) - (β*v₀') .* (4/(1+dot(β, β))^2)
    end
    function ∂H∂v(v)
      (data'*(data.*qⁿ)*v - (data'*qⁿ) .* (2*mⁿ)) .* 2.0
    end
    function ∂Hφ₀⁻¹P₀⁻¹∂β(β)
      ∂φ₀⁻¹P₀⁻¹∂β(β) * ∂H∂v(φ₀⁻¹P₀⁻¹(β))
    end
    function f(α)
      H(φ₀⁻¹P₀⁻¹(βⁿ-dⁿ.*α))
    end
    function dfdα(α)
      ∂Hφ₀⁻¹P₀⁻¹∂β(βⁿ-dⁿ.*α)'*(-dⁿ)
    end

    vⁿ=v₀;
    val_logFₙ = logFₙ(vⁿ, data=data);
    βⁿ = P₀φ₀(vⁿ);
    yⁿ = zeros(N);
    mⁿ = 0.0;
    hⁿ = 0.0;
    qⁿ = zeros(N);
    dⁿ = zeros(length(βⁿ));
    linesearch = BackTracking();

    for _ in 1:Loop
      if dot(vⁿ, v₀) < 0
        vⁿ = -vⁿ;
        βⁿ = P₀φ₀(vⁿ);
      end
      yⁿ = data * vⁿ;
      hⁿ = 1.144 * mad(yⁿ) * N^(-0.2);
      if hⁿ < 1e-10
        break;
      end
      mⁿ = uniMode(yⁿ);
      qⁿ .= ϕₕ.(yⁿ .- mⁿ, hⁿ);
      qⁿ .= qⁿ ./ sum(qⁿ);
      ξⁿ = ∂Hφ₀⁻¹P₀⁻¹∂β(βⁿ);
      dⁿ .= ξⁿ;
      αⁿ = dfdα(0.0) < -1e-5 ? linesearch(f, 1., f(0.0), dfdα(0.0))[1] : 0.0;
      βˢ = βⁿ - dⁿ.*αⁿ;
      val_logFₛ = logFₙ(φ₀⁻¹P₀⁻¹(βˢ), data=data);
      if val_logFₙ > val_logFₛ
        break;
      elseif norm(val_logFₛ - val_logFₙ) < ϵ
        βⁿ .= βˢ;
        vⁿ .= φ₀⁻¹P₀⁻¹(βⁿ);
        mⁿ = uniMode(data*vⁿ |> vec);
        val_logFₙ = val_logFₛ;
        break;
      else
        val_logFₙ = val_logFₛ;
        βⁿ .= βˢ;
        vⁿ .= φ₀⁻¹P₀⁻¹(βⁿ);
      end
    end

    (val_logFₙ, vⁿ)
  end


  kernelX = nullspace(X);
  cnt = size(kernelX)[2];
  if cnt > 0
    W[:,1:cnt] = kernelX;
    kernel_kernelX = nullspace(kernelX');
    projX = X*kernel_kernelX;
  else
    projX = X;
  end
  projd = size(projX)[2];
  cnt = cnt + 1;

  projW = zeros(projd, projd);
  L = zeros(projd, projd);
  vals = zeros(projd);
  M = min(N, 20);
  Xsample = view(projX, sample(1:N, M, replace=false), :);

  Vs = zeros(projd, round(Int, M*(M-1)*0.5))
  tmp = 0
  for i in 1:M-1
    Vs[:, (tmp+1):(tmp+(M-i))] = Xsample[:, (i+1):M] .- Xsample[:, i];
    tmp = tmp + (M-i);
  end
  Vs .= mapreduce(i->Vs[:,i]./norm(Vs[:,i]), hcat, round(Int, M*(M-1)*0.5));

  outs = zeros(N);
  outvals = zeros(N);
  for i in 1:round(Int, M*(M-1)*0.5)
    ys = projX * Vs[:,i];
    res = rcopy(R"covMcd($(ys), alpha=0.5)");
    wv = abs.(ys .- res[:center]) ./ res[:cov]
    outs = map(i -> wv[i] > outs[i] ? wv[i] : outs[i], 1:N);
  end
  outs = 1 ./ (outs .+ 1);
  outs = outs ./ sum(outs);

  meanv = projX' * outs;
  Y = projX .- meanv';
  # L .= eigfact(Symmetric(Y'*(Y.*outs)), 1:projd)[:vectors];
  L .= eigen(Symmetric(Y'*(Y.*outs)), 1:projd).vectors;
  vals = map(l -> logFₙ(L[:,l], data=projX), 1:projd);
  L .= L[:, sortperm(vals, rev=true)];
  vals .= vals[sortperm(vals, rev=true)];
  projW[:,1:projd] .= L[:,1:projd];

  vs = zeros(projd, Ng);
  ind = 1; # inds = projd-K+1; # (?)
  while ind < projd
    if ind != 1
      projW[:, ind] = orthvec(projW[:, ind], projW[:, 1:ind-1], L);
    end

    a = zeros(projd);
    a .= projW[:, ind];
    for k in 1:Nc
      for j in ind:min(ind+cld(projd,10), projd)
        vals_logF = zeros(Ng);
        θs = range(0, π/(2^(k-1)), length=Ng) |> collect;
        for l in 1:Ng
          vs[:,l] = a.*cos(θs[l]) + projW[:,j].*sin(θs[l]);
          vs[:,l] = vs[:,l] ./ norm(vs[:,l]);
          vals_logF[l] = logFₙ(vs[:,l], outs, data=projX);
        end
        θ₀ = θs[findmax(vals_logF)[2]];
        a = a.*cos(θ₀) + projW[:,j] .* sin(θ₀);
        a = a ./ norm(a);
      end
    end

    (vals[ind], projW[:,ind]) = ind == 1 ? searchcomp(a, data=projX) : searchcomp(a, subW=view(projW, :, 1:ind-1), data=projX);

    for j in ind+1:projd
      projW[:,j] .= orthvec(projW[:,j], projW[:, 1:j-1], Matrix(1.0I,projd,projd));
      vals[j] = logFₙ(projW[:, j], data=projX);
    end
    projW[:,ind+1:projd] .= projW[:, sortperm(vals[ind+1:projd], rev=true).+ind];
    ind = ind + 1;

  end

  if cnt > 1
    W[:,cnt:d] .= kernel_kernelX*projW;
  else
    W .= projW;
  end

  ms = map(i->uniMode(X*W[:,i]), 1:d);
  center = W * ms;
  loadings = W[:, d:-1:1];
  scores = (X .- ms') * loadings;

  return (center, loadings, scores);
end

function uniMode(xs::Vector)::Float64
  function ϕ(z::Float64)::Float64
    exp(-z*z*0.5) / sqrt(2π)
  end
  function ϕₕ(z::Float64, h::Float64)::Float64
    ϕ(z/h)/h
  end

  N = length(xs);
  h = max(1e-10, 1.144*mad(xs)*N^(-0.2));
  hsm = HSM(xs);
  ntn = NewtonforKDE(xs, init=hsm);

  if sum(ϕₕ.(xs .- hsm, h)) > sum(ϕₕ.(xs .- ntn, h))
    MEM(xs, init=hsm)
  else
    ntn
  end
end

function HSM(
  xs::Vector;
  isSorted::Bool=false
)::Float64
  N = length(xs);
  if N == 1
    return xs[1];
  elseif N == 2
    return 0.5*(xs[1] + xs[2]);
  else
    ys = isSorted ? xs : sort(xs);
    M = cld(N,2);
    w = ys[N] - ys[1];
    stid = 1;
    for id in 1:fld(N,2)+1
      if (ys[id+M-1] - ys[id]) < w
        stid = id;
        w = ys[id+M-1] - ys[id];
      end
    end

    return HSM(ys[stid:stid+M-1], isSorted=true)
  end
end

function NewtonforKDE(
  xs::Vector;
  init=nothing, Loop::Int64=100, ϵ::Float64=1e-10
)::Float64
  BORDER = xs .|> abs |> maximum;
  N = length(xs);
  h = max(1e-10, 1.144*mad(xs)*N^(-0.2));
  xⁿ = init==nothing ? HSM(xs) : init;
  xˢ = 0.0;

  function ϕ(z::Float64)::Float64
    exp(-z*z*0.5) / sqrt(2π)
  end
  function ϕₕ(z::Float64, h::Float64)::Float64
    ϕ(z/h)/h
  end
  function dLdx(x₀::Float64)::Float64
    -dot((-xs.+x₀)./h, ϕₕ.(-xs.+x₀, h))/(N*h)
  end
  function ddLdxx(x₀::Float64)::Float64
    -dot((-((-xs.+x₀)./h).^2).+1, ϕₕ.(-xs.+x₀, h) )/(N*h*h)
  end

  for _ in 1:Loop
    xˢ = xⁿ - dLdx(xⁿ) / ddLdxx(xⁿ);
    if abs(xˢ - xⁿ) < ϵ
      xⁿ = xˢ;
      break;
    elseif isnan(xˢ) || abs(xˢ) > BORDER
      xⁿ = 0.0;
      break;
    else
      xⁿ = xˢ;
    end
  end

  xⁿ
end

function MEM(xs::Vector;
  init=nothing, Loop::Int64=100, ϵ::Float64=1e-10
)::Float64
  function ϕ(z::Float64)::Float64
    exp(-z*z*0.5) / sqrt(2π)
  end
  function ϕₕ(z::Float64, h::Float64)::Float64
    ϕ(z/h)/h
  end

  if init == nothing
    init = HSM(xs);
  end

  N = length(xs);
  h = max(1e-10, 1.144*mad(xs)*N^(-0.2));
  xⁿ = init;
  for _ in 1:Loop
    q = ϕₕ.(xs .- xⁿ, h);
    q = q ./ sum(q);
    xˢ = dot(xs, q);
    if abs(xˢ - xⁿ) < ϵ
      xⁿ = xˢ;
      break;
    else
      xⁿ = xˢ;
    end
  end

  xⁿ
end

function orthvec(
  v::Vector, W::Array{Float64,2}, basis::Array{Float64,2};
  ϵ=1e-8
)::Vector
  kernelW = nullspace(W');
  d = length(v);
  ret = zeros(d);
  if norm(kernelW'*v) < ϵ
    for i in 1:d
      if norm(kernelW'*basis[:,i]) > ϵ
        ret .= kernelW * kernelW' * basis[:,i];
        break;
      end
    end
  else
    ret .= kernelW * kernelW' * v;
  end

  ret ./ norm(ret)
end