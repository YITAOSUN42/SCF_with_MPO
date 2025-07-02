using ITensors
using ITensorMPS
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using HDF5
using Quantics
include("2D_lattice.jl") 
 

function novel_potential(L,U)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) = begin
 
        xc = (x % 2^15) - 2^14
        yc = div(x, 2^15) - 2^14
     
        r1 = sqrt(2)/2 * xc + sqrt(2)/2 * yc
        r2 = -sqrt(2)/2 * xc + sqrt(2)/2 * yc
        return U+0.5*(cos( (pi*xc) /  ( 2^11*sqrt(3)) )  + cos( (pi*yc)  / ( 2^11*sqrt(3)) ) + cos( (pi*r1)  / ( 2^11*sqrt(3)) ) + cos( (pi*r2)  / ( 2^11*sqrt(3)) )+cos( (pi*xc) /  (2*  sqrt(5) ))  + cos( (pi*yc)  / ( 2* sqrt(5)) ) + cos( (pi*r1)  / ( 2* sqrt(5)) ) + cos( (pi*r2)  / (2* sqrt(5)) ))
    
    end
    
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return density_mpo 
end

function novel_initial_guess_up(L)
    
    xvals = range(0, (2^L - 1); length=2^L)
 
    f(x) =  (x +((div(x,L_chain))%2  ))%2
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return qtt,density_mpo,density_mps 
end

function novel_initial_guess_down(L)
    
    xvals = range(0, (2^L - 1); length=2^L)
 
    f(x) = (x+1+((div(x,L_chain))%2  ))%2
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return qtt,density_mpo,density_mps
end



function novel_hop_row(L,t)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) = 1 
    
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return density_mpo 
end

function novel_hop_coulmn(L,t)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) =  1 
    
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return density_mpo 
end

L_chain = 2^15
num_site = 2^30
 
t = 1
sites = siteinds("Qubit",30,conserve_qns=false); 

#An example shown the construction of the 8-fold symmetric super moire square lattice hamiltonian for Fig. 4 in the manuscript

let
    
    id = OpSum()
    for j = 1:Int(log2(num_site))
        id += 1,"Id",j
    end
    Id_op = MPO(id,sites); 
    u_mpo = novel_potential(30,8);
    qtt_den_old_up, initial_den_up,ini_mps_up = novel_initial_guess_up(Int(log2(num_site)));
    qtt_den_old_down, initial_den_down,ini_mps_down  = novel_initial_guess_down(Int(log2(num_site)));

    hop_intra =  novel_hop_row(Int(log2(num_site)),t);
    hop_inter =  novel_hop_coulmn(Int(log2(num_site)),t);
    intra_mpo =  intrachain_hopping(L_chain, hop_intra,num_site, sites);
    inter_mpo =  interchain_hopping_square(L_chain, hop_inter, num_site, sites);
    
    k_mpo = intra_mpo + inter_mpo 
    #initial MF Hamiltonian
    u_mpo_ini_up = apply(u_mpo, mpo_guess_up)
    u_mpo_ini_down = apply(u_mpo, mpo_guess_down)
    
    #half-filling
    ham_up = +(k_mpo, u_mpo_ini_down, -0.5 * u_mpo)
    ham_down = +(k_mpo, u_mpo_ini_up, -0.5 * u_mpo)

    return 
end