using LinearAlgebra

function RPCAOM(
  data;
  maxd=round(Int, 0.5*size(data,2)), MaxLoop=100, ϵ=1e-5
)::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
  (N, d) = size(data)
  ds = ones(N)
  D = diagm(ds)

  U=zeros(d, maxd)
  b=zeros(d)

  for loop in 1:MaxLoop
    mat = data' * (I - D*ones(N)*ones(N)'./dot(ones(N), D*ones(N))) * sqrt.(D)
    Uₛ = svd(mat).U[:,1:maxd]
    bₛ = data' * D * ones(N) ./ dot(ones(N), D*ones(N))
    dsₛ = [1.0/(2*norm( (I-Uₛ*Uₛ')*(data[i,:]-bₛ) )) for i in 1:N]
    Dₛ = diagm(dsₛ)

    if norm(U-Uₛ) < ϵ
      break
    end
    U .= Uₛ
    b .= bₛ
    ds .= dsₛ
    D .= Dₛ
  end

  scores = (data .- b')*U
  # Dict{Symbol, Any}(
  #   :PC => U,
  #   :b => b
  # )
  return (b, U, scores)
end
