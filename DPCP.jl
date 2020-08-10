using JuMP, GLPK, Statistics, LinearAlgebra, ProgressMeter
include("GeometricMedian.jl");

function DPCP(
  inpX::Array{Float64,2};
  progress = false,
  isCentered = false,
  scale=true,
  enableHybrid=true,
  maxd=0
)::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
  δ = 1e-8;
  ϵ = 1e-4;
  MaxLoop = 100;
  LargeM = 1e+12;

  (N, d) = size(inpX);
  gm = zeros(d);
  X = zeros(N, d);
  W = zeros(d,d);
  if !isCentered
    gm .= GM(inpX);
    X .= inpX .- gm';
  else
    X .= inpX;
  end
  if scale
    X .= Diagonal( map(i->1/norm(X[i,:]), 1:N) ) * X;
  end
  if progress
    meter = Progress(d);
  end

  function getMC(k::Int64)::Array{Float64,1}

    function _lp(wₗ::Array{Float64,1})::Array{Float64,1}
      model = Model(with_optimizer(GLPK.Optimizer));
      @variable(model, m⁺[1:N]);
      @constraint(model, m⁺.>=0);
      @variable(model, m⁻[1:N]);
      @constraint(model, m⁻.>=0);
      @variable(model, w[1:d]);
      @constraint(model, sum(w[l]*wₗ[l] for l in 1:d) == 1);
      for i in 1:N
        @constraint(model, sum(X[i,l]*w[l] for l in 1:d) == m⁺[i] - m⁻[i]);
      end
      for j in 1:k-1
        @constraint(model, sum(w[l]*W[l,j] for l in 1:d) == 0)
      end
      @objective(model, Min, sum(m⁺[i] + m⁻[i] for i in 1:N));
      optimize!(model);
      return value.(w);
    end

    function _wls(wₗ::Array{Float64,1})::Array{Float64,1}
      XWX = X'*(X.*(1 ./ max.(δ, abs.(X*wₗ)) ));
      V = hcat(wₗ, W[:,1:k-1]);
      e₁ = Matrix(1.0I, k, 1);
      return (
        k==1 ?
          XWX \ wₗ :
          (XWX \ V) * ( (V'*(XWX \ V)) \ e₁ )
        ) |> vec;
    end

    wₙ = vec(eigen(Symmetric((X'*X)./N + (W[:,1:k-1]*W[:,1:k-1]').*LargeM), 1:1).vectors)
    ŵ = zeros(d);
    for l in 1:MaxLoop
      if enableHybrid
        ŵ .= _wls(wₙ);
        if norm(X*(ŵ./norm(ŵ)),1) > norm(X*wₙ,1)
          ŵ .= _lp(wₙ);
        end
      else
        ŵ .= _lp(wₙ);
      end
      wₛ = ŵ./norm(ŵ);
      Jₙ = norm(X*wₙ, 1);
      Jₛ = norm(X*wₛ, 1);
      ΔJ = (Jₙ - Jₛ)/Jₙ;
      wₙ .= wₛ;
      if ΔJ < ϵ
        break;
      end
    end
    return wₙ;
  end

  for j in 1:(maxd!=0 ? maxd : d)
    W[:,j] .= getMC(j);
    if progress
      next!(meter; showvalues=[(:finished, "$(j)-th MC")])
    end
  end

  loadings = W[:,d:-1:1];
  scores = X*loadings;
  return (gm, loadings, scores);
end
