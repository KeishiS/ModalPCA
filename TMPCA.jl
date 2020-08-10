# Trimmed Median PCA
using LinearAlgebra, Distributions, StatsBase, Statistics, Random

function TMPCA(
  inpX::Array{Float64,2}; maxd = 0,
  jₛ = 100, kₛ = 10,
  kₘ = 10, s = 2,
  c = 3, cₘ = 2,
  cᵧ = 2, α = 0.75,
  ϵ = 0.01, p = 5
)::Tuple{Array{Float64,1},Array{Float64,2}}
  MaxLoop = 100;
  (N, d) = size(inpX);
  h = round(Int, (N + d + 1) / 2);
  nₛ = max(h + 1, round(Int, N * 0.7));
  if maxd == 0
    maxd = d;
  end

  mₑ = map(l->median(inpX[:,l]), 1:d);
  MAD = map(i->norm(inpX[i,:] - mₑ), 1:N) |> median;
  X = (inpX .- mₑ') ./ MAD;

  function weightedMCMPCA(H::Array{Int64,1})::Tuple{Array{Float64,1},Array{Float64,2}}
    function weight(x::Array{Float64,1}, m::Array{Float64,1})::Float64
      return norm(x - m) / c >= 1 ? 0 : (1 - (norm(x - m) / c)^2)^2
    end

    # geometric medianを得るための反復処理
    m̄ = zeros(d);
    ms = zeros(d, length(H) + 1);
    m̄s = zeros(d, length(H) + 1);
    ms[:,1] .= mₑ;
    m̄s[:,1] .= mₑ;
    ξ = 1;
    for l in 1:MaxLoop
      m̄ .= m̄s[:,1];
      ws = map(i->(i, weight(X[i,:], m̄)), H) |> Dict;
      for (i, value) in enumerate(shuffle(H))
        xᵢ = X[value,:];
        wᵢ = ws[value];
        γ = cₘ / ((ξ + i)^α);
        ms[:,i + 1] .= ms[:,i] + (xᵢ - ms[:,i]) .* ( wᵢ * γ / norm(xᵢ - ms[:,i]) );
        m̄s[:,i + 1] .= m̄s[:,i] - (m̄s[:,i] - ms[:,i + 1]) ./ (ξ + i + 1);

        ξ += 1;
      end
      if norm(m̄s[:,1] - m̄s[:,length(H) + 1]) < ϵ
        break;
      else
        ms[:,1] .= ms[:,length(H) + 1];
        m̄s[:,1] .= m̄s[:,length(H) + 1];
      end
    end
    m̄ .= m̄s[:,1];
    ws = map(i->(i, weight(X[i,:], m̄)), H) |> Dict;

    # MCMを得るための反復処理
    Γs = zeros(d, d, length(H) + 1);
    Γ̄s = zeros(d, d, length(H) + 1);
    Γs[:,:,1] .= ((mean(X[H,:], dims = 1) |> vec) - m̄) * ((mean(X[H,:], dims = 1) |> vec) - m̄)';
    Γ̄s[:,:,1] .= Γs[:,:,1];
    ξ = 1;
    for l in 1:MaxLoop
      for (i, value) in enumerate(shuffle(H))
        xᵢ = X[value,:];
        wᵢ = ws[value];
        γ = cᵧ / ((ξ + i)^α);
        Γs[:,:,i + 1] .= Γs[:,:,i] + ( (xᵢ - m̄) * (xᵢ - m̄)' - Γs[:,:,i] ) .* (wᵢ * γ / norm((xᵢ - m̄) * (xᵢ - m̄)' - Γs[:,:,i]));
        Γ̄s[:,:,i + 1] .= Γ̄s[:,:,i] - (Γ̄s[:,:,i] - Γs[:,:,i + 1]) ./ (ξ + i + 1);

        ξ += 1;
      end

      if norm(Γ̄s[:,:,1] - Γ̄s[:,:,length(H) + 1]) < ϵ || reduce(|, isnan.(Γ̄s[:,:,length(H) + 1]))
        break;
      else
        Γs[:,:,1] .= Γs[:,:,length(H) + 1];
        Γ̄s[:,:,1] .= Γ̄s[:,:,length(H) + 1];
      end
    end
    Γ = Γ̄s[:,:,length(H) + 1];

    return (m̄, Γ);
  end

  # H...subset of {1,2,...N}
  ## H_out...updated subset of {1,2...N}
  function cstep(H::Array{Int64,1}, Ps::Array{Float64,3}, k::Int64; enableConvergence = false)::Array{Float64,3}
    # データの部分集合を与えると，そこからweightedMCMPCAで部分空間を求めて
    # そこへそのデータを射影した際の残差和を返す関数
    function subsetToSumofResiduals(subH::Array{Int64,1}, plane::Array{Float64,2})::Float64
      Proj = I - plane * plane';
      return mapreduce(i->norm(Proj * X[i,:]), +, subH);
    end

    ## Ps=zeros(d,maxd,numPlane) ... j個の初期planeを格納するための配列
    ## Ps[:,1,l], Ps[:,2,l]...Ps[:,maxd,l] は主成分ベクトル順に並んでいる．
    ## [初期準備]
    numPlane = size(Ps)[3];
    Hs = zeros(Int64, h, numPlane);
    for np in 1:numPlane
      Proj = I - Ps[:,:,np] * Ps[:,:,np]';
      resnorms = map(i->(i, norm(Proj * X[i,:])), H);
      sort!(resnorms, by = x->x[2]);
      Hs[:,np] .= map(i->resnorms[i][1], 1:h);
    end

    areConverged = fill(false, numPlane);
    for loop in 1:(enableConvergence ? MaxLoop : s)
      for pid in 1:numPlane
        if !areConverged[pid]
          # Hs[:,pid]の更新を行う
          ## weightedMCMPCAでPs[:,:,pid]の更新
          (_, retΓ̄) = weightedMCMPCA(Hs[:,pid]);

          ## Hの全データをPs[:,:,pid]の張る部分空間へ射影し残差を計算
          eigret = retΓ̄ |> Symmetric |> eigen;
          inds = sortperm(eigret.values, rev = true)[1:maxd];
          Proj = I - eigret.vectors[:,inds] * eigret.vectors[:,inds]';
          resnorms = map(i->(i, norm(Proj * X[i,:])), H);
          sort!(resnorms, by = x->x[2]);
          newHs = map(i->resnorms[i][1], 1:h);

          ## Hs[:,pid]の更新
          ## Hs[:,pid]に変化がなければareConverged[pid]=trueとする
          if norm(Ps[:,:,pid] - eigret.vectors[:,inds]) < ϵ
            areConverged[pid] = true;
          else
            Hs[:,pid] .= newHs;
            Ps[:,:,pid] .= eigret.vectors[:,inds];
          end
        end
      end
    end

    # numPlane個のPlaneからk個のPlaneを選定
    sums = map(i->subsetToSumofResiduals(Hs[:,i], Ps[:,:,i]), 1:numPlane);
    inds = sortperm(sums)[1:k];
    return Ps[:,:,inds];
  end

  # subs = nₛ \times p matrix
  subs = map(i->sample(1:N, nₛ, replace = false), 1:p);
  subPs = zeros(d, maxd, kₛ * p);
  for l in 1:p
    initPs = zeros(d, maxd, jₛ);
    for r in 1:jₛ
      initS = sample(subs[l], maxd, replace = false);
      while rank(X[initS,:]' * X[initS,:]) < maxd
        initS = vcat(initS, sample(setdiff(subs[l], initS), 1));
      end
      eigret = X[initS,:]' * X[initS,:] |> Symmetric |> (A->eigen(A, d - maxd + 1:d));
      initinds = sortperm(eigret.values, rev = true)[1:maxd];
      initPs[:,:,r] .= eigret.vectors[:,initinds];
    end
    subPs[:,:,(l - 1) * kₛ + 1:l * kₛ] .= cstep(subs[l], initPs, kₛ);
  end

  resultPs = cstep(reduce(union, subs), subPs, kₘ);
  bestPs = cstep(collect(1:N), resultPs, 1, enableConvergence = true)[:,:,1];

  return (mₑ, bestPs);
end