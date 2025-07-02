using ITensors
using ITensorMPS
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using HDF5
using Quantics
include("2D_lattice.jl") 
 
#Not used in this case
function novel_potential(L,U)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) = begin
 
        xc = (x % 2^15) - 2^14
        yc = div(x, 2^15) - 2^14
     
        r1 = sqrt(2)/2 * xc + sqrt(2)/2 * yc
        r2 = -sqrt(2)/2 * xc + sqrt(2)/2 * yc
        return U+0.1*(cos( (pi*xc) /  ( 2^11*sqrt(3)) )  + cos( (pi*yc)  / ( 2^11*sqrt(3)) ) + cos( (pi*r1)  / ( 2^11*sqrt(3)) ) + cos( (pi*r2)  / ( 2^11*sqrt(3)) )+cos( (pi*xc) /  (2*  sqrt(5) ))  + cos( (pi*yc)  / ( 2* sqrt(5)) ) + cos( (pi*r1)  / ( 2* sqrt(5)) ) + cos( (pi*r2)  / (2* sqrt(5)) ))
    
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

#Neel order initial guess
function novel_initial_guess_up(L)
    
    xvals = range(0, (2^L - 1); length=2^L)
 
    f(x) =  (x)%2
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
 
    f(x) = (x+1 )%2
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return qtt,density_mpo,density_mps
end



C1 = π / (2^12*√3)
C2 = π / (4*√2)

function novel_hop_row(L,t,L_chain)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) = let uc = div(x % L_chain, 2), lc = div(x, L_chain)
        iseven(x % L_chain) ?
            ((1 + 0.2 * sin(C1 * ((3uc + 1)/2))) + ( 0.2 * sin(C2 * ((3uc + 1)/2)))) :
            (1 + 0.2 * sin(C1 * ((2.5 + 3uc)/4 + (iseven(uc) ? (3/8 + 1.5lc) : (-3/8 - 1.5lc))))) + ( 0.2 * sin(C2 * ((2.5 + 3uc)/4 + (iseven(uc) ? (3/8 + 1.5lc) : (-3/8 - 1.5lc)))))
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

function novel_hop_1(L,t,L_chain)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) = let uc = div(x % L_chain, 2), lc = div(x, L_chain)
        (iseven(uc) || isodd(x)) ? 0.0 :
        1 + 0.2 * sin(C1 * (0.75uc - 1.5lc - 1.25)) + 0.2 * sin(C2 * (0.75uc - 1.5lc - 1.25))
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
    
function novel_hop_2(L,t,L_chain)
    
    xvals = range(0, (2^L - 1); length=2^L)
    
    f(x) = let uc = div(x % L_chain, 2), lc = div(x, L_chain)
        (iseven(uc) || iseven(x)) ? 0.0 :
        1 + 0.2 * sin(C1 * ((2.5 + 3uc)/4 + (9 + 12lc)/8)) + 0.2 * sin(C2 * ((2.5 + 3uc)/4 + (9 + 12lc)/8))
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

L_chain = 2^16
num_site = 2^31
 
t = 1
 
sites = siteinds("Qubit",31,conserve_qns=false);

#An example shown the construction of the super moire graphene used in the SM
let
       
    id = OpSum()
    for j = 1:Int(log2(num_site))
        id += 1,"Id",j
    end

    Id_op = MPO(id,sites); 
    u_mpo = 5.5*(Id_op/31);
    qtt_den_old_up, initial_den_up,ini_mps_up = novel_initial_guess_up(Int(log2(num_site)));
    qtt_den_old_down, initial_den_down,ini_mps_down  = novel_initial_guess_down(Int(log2(num_site)));

    hop_intra = novel_hop_row(Int(log2(num_site)),t,L_chain);
    hop_1 = novel_hop_1(Int(log2(num_site)),t,L_chain);
    hop_2 = novel_hop_2(Int(log2(num_site)),t,L_chain);
    break_mpo = break_chain(L_chain, num_site, sites)
    real_hop_2 = apply(break_mpo,hop_2);
    intra_mpo = intrachain_hopping(L_chain, hop_intra,num_site, sites);
    inter_mpo = interchain_hopping_honeycomb(L_chain, num_site, real_hop_2,hop_1, sites); 
    
    k_mpo = intra_mpo + inter_mpo 
    #initial MF Hamiltonian
    u_mpo_ini_up = apply(u_mpo, mpo_guess_up)
    u_mpo_ini_down = apply(u_mpo, mpo_guess_down)
    
    #half-filling
    ham_up = +(k_mpo, u_mpo_ini_down, -0.5 * u_mpo)
    ham_down = +(k_mpo, u_mpo_ini_up, -0.5 * u_mpo)
end