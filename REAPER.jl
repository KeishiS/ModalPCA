using LinearAlgebra, Statistics
include("GeometricMedian.jl");

function REAPER(
  inpX::Array{Float64,2},
  k::Int64;
  isCentered=false
)::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
  δ=1e-10;
  ϵ=1e-10;
  Loop=100;
  (N,d)=size(inpX);
  X=zeros(N,d)


  function WLS(data, β, k)
    C=data'*(data .* β);
    (λs, U) = C |> Symmetric |> eigen;
    U.=U[:,sortperm(λs, rev=true)];
    λs.=sort(λs, rev=true);

    if k<d && abs(λs[min(k+1,d)]) < ϵ
      return U[:,1:k]*U[:,1:k]';
    else
      let
        sumval=sum(1 ./ λs[1:k]);
        θ=0.;
        for i in k+1:d
          sumval += 1/λs[i];
          if i==d
            θ = (i-k)/sumval;
            break;
          elseif λs[i] > (i-k)/sumval >= λs[i+1]
            θ = (i-k)/sumval;
            break;
          end
        end
        λs.=map(i->λs[i]>θ ? 1-θ/λs[i] : 0, 1:d);
        end
      return U*(U' .* λs);
    end
  end

  gm = zeros(d);
  if !isCentered
    gm .= GM(inpX);
  end
  X .= inpX .- gm';

  αⁿ=Inf;
  β=ones(N);
  Pⁿ=zeros(d,d);
  A=zeros(d,d);
  for l in 1:Loop
    Pⁿ.=WLS(X, β, k);
    A.=Matrix(1.0I,d,d)-Pⁿ;
    αˢ=sum(map(i->β[i]*norm(A*X[i,:])^2, 1:N));
    β.=map(i->1/max(δ, norm(A*X[i,:])), 1:N);

    if abs(αˢ-αⁿ) < ϵ
      break;
    else
      αⁿ=αˢ;
    end
  end

  (λs, U) = eigen(Symmetric(Pⁿ));
  U.=U[:,sortperm(λs, rev=true)];
  scores = X*U[:,1:k]

  return (gm, U[:,1:k], scores);
end
