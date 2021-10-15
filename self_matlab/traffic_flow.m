% Define graph
Nodes = [0,0 ; 0,2 ; 1,1 ; 2,1 ; 3,0 ; 3,2];
Edges = [1,3 ; 2,3 ; 3,4 ; 4,5 ; 4,6];

% Discretization parameters
ne = 100;
nt = 1000;

% Some numbers
nei = ne-2;
n_edges = size(Edges, 1);
n_nodes = size(Nodes, 1);
n_inner = nei*n_edges;
n_dofs = n_inner+n_nodes;

% Nodes = [0,0 ; 1,0 ];
% Edges = [1,2];
InflowNodes = [1];
OutflowNodes = [2];

% incidence matrix
N = sparse(kron(1:n_edges,[1,1]), ...
           reshape(Edges',1,[]), ...
           kron(ones(1,n_edges),[-1,1]));

alpha = [0.9, 0.3, 0, 0, 0, 0];
beta = [0, 0, 0, 0, 0.8, 0.1];

% Define inflow and outflow nodes
InflowNodes = [1, 2];
OutflowNodes = [5, 6];
CouplingNodes = [3, 4];

% Model parameters
T = 10;
L = [1];
eps = 1.e-2;
eta = 1;

% flux function
%f = @(u)(-0.25*(1-2*u).^2); % Our model
f = @(u)(u.*(1-u)); % Burgers equation

idxi = 1:nei;
idxn = 1:n_nodes;
n_coupling = 1;
he = L/ne;
tau = T/nt;
x = he/2:he:L(1);

% Inflow condition
%g = @(t)(0.4*[mod(floor(t+1),3);mod(floor(t+2),3)]);
g = @(t)(0.5+0.5*[sin(pi*t);sin(pi*(t+1))]);
%g = @(t)(0*t+0.9);

% Initial condition
u = zeros(n_dofs,nt);
%u(:,1) = max(0,min(1,2-2*abs(x-1)));
%u(:,1) = 0.9*(x>0.3).*(x<0.6); % + 0.45*(x<=2).*x;

S = sparse(n_dofs, n_dofs);
M = sparse(n_dofs, n_dofs);

% Mass and stiffness matrix for test functions in edge interior
for k=0:size(Edges, 1)-1
    
    % Mass matrix
    M_II = he*spdiags(ones(nei,1), 0, nei, nei);
    
    % Diffusion term (interior intervals only)
    D_II = 1/he*spdiags([-ones(nei,1), 2*ones(nei,1), -ones(nei,1)], [-1,0,1], nei, nei);
    D_VI = sparse([1,nei], Edges(k+1,:)', -1/he*[1,1], nei, n_nodes);
    
    % Sort into global matrix
    S(k*nei+idxi, k*nei+idxi) = M_II+tau*eps*D_II;
    S(k*nei+idxi, n_inner+idxn) = S(k*nei+idxi, n_inner+idxn) + tau*eps*D_VI;
    S(n_inner+idxn, k*nei+idxi) = S(n_inner+idxn, k*nei+idxi) + tau*eps*D_VI';
    
    M(k*nei+idxi, k*nei+idxi) = M_II;
end

% Mass and stiffness matrix for test functions in vertec patch
M_VV = he*spdiags(sum(abs(N),1)', 0, n_nodes, n_nodes);
D_VV = 1/he*spdiags(sum(abs(N),1)', 0, n_nodes, n_nodes);

S(n_inner+idxn,n_inner+idxn) = M_VV+tau*eps*D_VV;
M(n_inner+idxn,n_inner+idxn) = M_VV;

% Incorporate Dirichlet boundary conditions
% n_inflow = size(InflowNodes,1);
% for i=InflowNodes
%     S(n_inner+i, :) = sparse(1, n_inner+i, he, 1, n_dofs);
% end

% Loop over time steps
figure(1);
for k=2:nt
    fprintf("Timestep %d/%d\n", k, nt);
    
    % Assemble right-hand side
    t = (k-1)*tau;
    g_ = g(t);
    
    % Assemble advection term
    F = zeros(n_dofs,1);
    
    uk = u(:,k-1);
    
    % test functions in the interior of each edge
    for i=1:n_edges
        
        F_ = zeros(nei,1);
        
        % All entries on the edge
        u_ = [uk(n_inner+Edges(i,1)) ; uk((i-1)*nei+idxi) ; uk(n_inner+Edges(i,2))];
        
        % Flux at right and left end of the interval
        F_ = F_ + (f(u_(3:end)) + f(u_(2:end-1)))/2 + 0.5*eta*(u_(2:end-1) - u_(3:end));
        F_ = F_ - (f(u_(2:end-1)) + f(u_(1:end-2)))/2 + 0.5*eta*(u_(2:end-1) - u_(1:end-2));
        
        % Insert into global flux vector
        F((i-1)*nei+idxi) = F_;
        
    end
    
    % test functions in vertices
    for i=1:n_nodes
        u_v = uk(n_inner+i);
       
        for j=1:n_edges
            n = N(j,i);
            
            if(n == -1)
                u_e = uk((j-1)*nei+1);
                F(n_inner+i) = F(n_inner+i) ...
                    + (f(u_v) + f(u_e))/2 ...
                    + 0.5*eta*(u_v - u_e);
            elseif(n == 1)
                u_e = uk((j-1)*nei+nei);
                F(n_inner+i) = F(n_inner+i) ...
                    - (f(u_v) + f(u_e))/2 ...
                    + 0.5*eta*(u_v - u_e);
            end
        end
        
        % JAN: Hier weiter...
        if sum(int8(InflowNodes==i)) > 0 % At inflow node
            F(n_inner+i) = F(n_inner+i) -alpha(i)*(1-u_v);
        elseif sum(int8(OutflowNodes==i)) > 0 
                F(n_inner+i) = F(n_inner+i) + beta(i)*u_v;
        end
        
    end
    
    rhs = M*u(:,k-1) - tau*F;
    
    % Incorporate Dirichlet BC
%     for i=InflowNodes
%         rhs(n_inner+i) = he*g_(i);
%     end
    
    % Solve equation system
    u(:,k) = S\rhs;
    
    % Plot current timestep
    u_ = u(:,k);
        
    % Plotting (3D)
    for i=1:n_edges
        xs = linspace(Nodes(Edges(i,1),1), Nodes(Edges(i,2),1), ne);
        ys = linspace(Nodes(Edges(i,1),2), Nodes(Edges(i,2),2), ne);
        zs = [u_(n_inner+Edges(i,1)) ; u_((i-1)*nei+idxi) ; u_(n_inner+Edges(i,2))]';
        
        % Plot function
        plot3(xs,ys,zs,'b-','LineWidth',1);
        hold on;
        zlim([0,1.2]);
        view(10,20);
        
        % Plot graph
        plot3(xs,ys,zeros(size(xs)),'r-','LineWidth',2);
        hold on;
    end
    hold off;
    
    pause(0.01)
end