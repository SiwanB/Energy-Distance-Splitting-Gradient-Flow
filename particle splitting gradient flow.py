import numpy as np
import torch
import pylab as plt
from tqdm import tqdm


plt.ion()
plt.show()


#%%
if torch.cuda.is_available():
    torch.cuda.empty_cache()

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    


#%%



def gen_sphere(P,dim):
    vects = torch.normal(torch.zeros((P, dim),device=device))
    norms = torch.linalg.norm(vects, dim=1, keepdim=True)
    vects /= norms
    return vects


def proj_tens(X,VECTS):
    return torch.einsum('id,pd->ip',X,VECTS)

    
def D(A,B):
    scal = torch.einsum('id,jd->ij',A,B)
    X2 = torch.einsum('id->i',A**2)
    Y2 = torch.einsum('id->i',B**2)
    return torch.sqrt(torch.relu(X2.unsqueeze(1) + Y2 -2*scal))

def ED(A,B):
    n = A.size()[0]
    m = B.size()[0]
    return 2*torch.sum(D(A,B))/(n*m)-torch.sum(D(A,A))/(n**2) - torch.sum(D(B,B))/(m**2)


def ed_1d_serie(x,y):
    N = x.shape[0]
    M = y.shape[0]
    P = x.shape[1]
    x_tri = torch.sort(x,dim=0)[0]
    y_tri = torch.sort(y,dim=0)[0]
    
    z = torch.cat((x,y),dim=0)
    sig_inv = torch.argsort(z,dim=0)
    z = torch.gather(z,0,sig_inv)
    ind_y = torch.zeros((N+M,P),device=device)
    ind_y[N:N+M] = 1
    ind_y = torch.gather(ind_y,0,sig_inv)
    a_y = 2*torch.cumsum(ind_y,dim=0)-M
    b_y = 2*torch.cumsum(ind_y*z,dim=0)-torch.sum(y,dim=0)
    
    
    
    a_n = 2*torch.cumsum(torch.ones(N,device=device),dim=0)-N
    b_n = 2*torch.cumsum(x_tri,dim=0)-torch.sum(x,dim=0)
    a_m = 2*torch.cumsum(torch.ones(M,device=device),dim=0)-M
    b_m = 2*torch.cumsum(y_tri,dim=0)-torch.sum(y,dim=0)
    
    return 2*torch.sum((1-ind_y)*(z*a_y-b_y),dim=0)/(N*M) - torch.sum((x_tri*a_n.unsqueeze(1)-b_n),dim=0)/(N*N) - torch.sum((y_tri*a_m.unsqueeze(1)-b_m),dim=0)/(M*M)


def sliced_factor(d):
    '''
    compute the scaling factor of sliced MMD
    '''
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0:
        for j in range(1,k+1):
            fac=2*fac*j/(2*j-1)
    else:
        for j in range(1,k+1):
            fac=fac*(2*j+1)/(2*j)
        fac=fac*np.pi/2
    return fac

def ED_sliced(A,B,P,facteur):
    vects = gen_sphere(P,dim)
    x,y = proj_tens(A,vects),proj_tens(B,vects)
    return facteur*torch.mean(ed_1d_serie(x, y))


def Energy_Based_ED(A,B,P,facteur,f = torch.exp):
    n = A.shape[0]
    vects = gen_sphere(P,dim)
    x,y = proj_tens(A,vects),proj_tens(B,vects)
    ed_tableau = facteur*ed_1d_serie(x, y)
    return torch.sum(ed_tableau*torch.softmax(ed_tableau,dim=0))
    #return torch.mean(ed_tableau**2 + ed_tableau)
    




def split_measure(X, eq_type = 10**-7):
    n = X.shape[0]
    res = torch.zeros((n+1,dim),device = device)
    X_copie = X.detach().clone()
    
    res[:n,:] = X_copie.detach().clone()
    
    ind = np.random.randint(0,n)
    
    value = X_copie[ind,:]
    
    res[ind,:] = value.detach().clone() + eq_type*torch.randn(dim,device=device)
    
    res[n,:] = value.detach().clone() + eq_type*torch.randn(dim,device=device)
    
    return res
    



#%%



M = 250

N_max = 5000

error = 0   # Erreur numérique acceptée pour l'évaluation de la best loss



t_split = 3   # Nb d'iters sans progrès autorisé avant de splitter

eq_type = 10**-7  # ecart type de la gaussienne utilisée pour splitter



dim = 3
P = 10 # Nombre de directions de projections pour les distances slicées








c = 1      # Paramètre Polyak step, garder c=1, en cas de pb c=2 ou c>1.
eps = 10**-8  # Paramètre Polyak step, pour éviter la division par zéro.



nb_iter = 10000



nb_affich = 0








facteur = sliced_factor(dim)

#%%

#Y = torch.cat([(0.5-torch.randn((M//2,dim),device=device)),15+(0.5-torch.randn((M//2,dim),device=device))])
Y = torch.cat([8+(0.5-torch.randn((M//3,dim),device=device)),(0.5-torch.randn((M//3,dim),device=device)),8+(0.5-torch.randn((M//3,dim),device=device))])
Y[:M//3,1:] = -Y[:M//3,1:]



#Y = (2*(0.5-torch.rand((M,dim),device=device)) + avg)
# Y = torch.randn((M,dim),device=device)
# Y = 2+(1-2*torch.rand((M,dim),device=device))


# norm = torch.norm(Y,dim=1,keepdim = True)
# Y = (dim**0.5)*Y/norm
#Y = torch.normal(torch.zeros((M,dim),device=device))-1
#Y[:,2+dim//2:] = 0
#Y[:,1]=0

X = torch.zeros((1,dim),device=device)

X.requires_grad_(True)



#%%


Times = np.unique(np.round(2**(np.linspace(0,np.log(nb_iter-1)/np.log(2),nb_affich))))

Times = np.round(np.linspace(0,nb_iter-1,nb_affich))



LOSS = []
LOSS_N = []
ALPHA = []
LOSS_COMP = []
lg = []
nb_part = []
GRAD = []
B_LOSS = []
B_LOSS_COURANT = []

#%%
with torch.no_grad():
    x_min = torch.min(torch.min(X[:,0]),torch.min(Y[:,0])).cpu().detach()
    x_max = torch.max(torch.max(X[:,0]),torch.max(Y[:,0])).cpu().detach()
    y_min = torch.min(torch.min(X[:,1]),torch.min(Y[:,1])).cpu().detach()
    y_max = torch.max(torch.max(X[:,1]),torch.max(Y[:,1])).cpu().detach()
    y_min=x_min
    y_max=x_max
    delta_x = 0.1*(x_max-x_min)
    delta_y = 0.1*(y_max-y_min)
    
    X_affich = X.cpu().detach()
    Y_affich = Y.cpu().detach()
    
    plt.figure()
    plt.plot(X_affich[:,0],X_affich[:,1],linestyle='',marker=',')

    plt.plot(Y_affich[:,0],Y_affich[:,1],linestyle='',marker=',')
    #plt.axis((x_min-delta_x,x_max+delta_x,y_min-delta_y,y_max+delta_y))
    
    plt.show()
    #pause()
x_min,x_max,y_min,y_max = -1,1,-1,1



#%%


ind_split = 0

compteur = 1


for i in tqdm(range(nb_iter)):
        
    X.grad = None
    
    
    
    ### CHOIX DE LA LOSS
    
    
    #loss = ED(X,Y)
    #loss = ED_sliced(X,Y,P,facteur)
    loss = Energy_Based_ED(X,Y,P,facteur)
    
    
    #############################################
    
    
    
    loss.backward()
    
    if i==0:
        with torch.no_grad():
            best_loss = loss
            best_loss_courant = loss
    
    with torch.no_grad():
        
        alpha_k = loss/(c*torch.norm(X.grad)**2 + eps)
        
        #alpha_k = (loss - 0.9*best_loss_courant)/(c*torch.norm(X.grad)**2 + eps)
        
        X -= alpha_k*X.grad
        
        LOSS.append(loss.cpu().detach())
        ALPHA.append(alpha_k.cpu().detach())
        
        
        if loss<best_loss + error:
            best_loss = loss.detach()
        
        if loss < best_loss_courant + error:
            best_loss_courant = loss.detach()
            ind_split = 0
        else:
            ind_split += 1
        #Plotting
        if i in Times:
            X_courant = X.cpu().detach()
            plt.figure()
            plt.plot(X_courant[:,0],X_courant[:,1],linestyle='',marker=',')

            plt.plot(Y_affich[:,0],Y_affich[:,1],linestyle='',marker=',')
            plt.title(str(i))
            #plt.axis((x_min-delta_x,x_max+delta_x,y_min-delta_y,y_max+delta_y))
            plt.show()
            
        
        #Comp with generic 'Y-like' measure
        Y_2 = torch.cat([8+(0.5-torch.randn((M//3,dim),device=device)),(0.5-torch.randn((M//3,dim),device=device)),8+(0.5-torch.randn((M//3,dim),device=device))])
        Y_2[:M//3,1:] = -Y_2[:M//3,1:]
            
        #Y_2 = 2+(1-2*torch.rand((M,dim),device=device))



        ### Attention :  A modifier (mais ne change pas l'algo)
        
        #LOSS_COMP.append(ED(X, Y_2).cpu().detach())
        #LOSS_COMP.append(ED_sliced(X, Y_2, P, facteur).cpu().detach())
        LOSS_COMP.append(Energy_Based_ED(X, Y_2, P, facteur).cpu().detach())

        
        nb_part.append(compteur)
        
        GRAD.append(torch.norm(X.grad).cpu().detach())
        
        
        #Splitting of the particles
        if compteur<N_max and ind_split > t_split:
            
            LOSS_N.append(loss.cpu().detach())
            lg.append(compteur)
            
            X = split_measure(X,eq_type)
            
            
            ### Attention : a modifier !!!
            
            #loss = ED(X,Y)
            #loss = ED_sliced(X,Y,P,facteur)
            loss = Energy_Based_ED(X,Y,P,facteur)
            
            best_loss_courant = loss.detach()
            
            X.requires_grad_(True)
            compteur += 1
            ind_split = 0
        
        B_LOSS.append(best_loss.cpu().detach())
        B_LOSS_COURANT.append(best_loss_courant.cpu().detach())
            
            
            
#%%

plt.figure()
plt.plot(np.arange(0,nb_iter),GRAD)
plt.title('gradient norm')

plt.figure()
plt.loglog(np.arange(0,nb_iter),GRAD)
plt.title('gradient norm loglog')

#%%

plt.figure()
plt.loglog(np.arange(0,nb_iter),B_LOSS)
plt.loglog(np.arange(0,nb_iter),B_LOSS_COURANT)
plt.title('best loss et best loss courant')

#%%

plt.figure()
plt.plot(np.arange(0,nb_iter),nb_part)
plt.title('nb particles')

plt.figure()
plt.plot(lg,LOSS_N)
plt.title('loss en fct de N')


plt.figure()
plt.plot(lg,LOSS_N)
plt.yscale('log')
plt.title('loss en fct de N log scale')


plt.figure()
plt.loglog(lg,LOSS_N)
plt.title('loss en fct de N loglog scale')


plt.figure()
plt.plot(np.arange(0,nb_iter),LOSS,label='loss1')
plt.plot(np.arange(0,nb_iter),LOSS_COMP,label='loss1 gen distr')
plt.title('plot loss')
plt.legend()



plt.figure()
plt.plot(np.arange(0,nb_iter),LOSS,label='loss1')
plt.plot(np.arange(0,nb_iter),LOSS_COMP,label='loss1 gen distr')
plt.yscale('log')
plt.title('plot loss log scale')
plt.legend()



plt.figure()
plt.loglog(np.arange(0,nb_iter),LOSS,label='loss1')
plt.loglog(np.arange(0,nb_iter),LOSS_COMP,label='loss1 gen distr')
plt.legend()
plt.title('plot loss loglog scale')


plt.figure()
plt.plot(np.array(ALPHA),label='loss1')
plt.yscale('log')
plt.title('alpha log scale')
plt.legend()


plt.figure()
plt.loglog(np.array(ALPHA),label='loss1')
plt.title('alpha loglog scale')

plt.legend()




























    