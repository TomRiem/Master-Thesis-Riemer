% Define graph
Nodes = [[0,0],[1,0]];
Edges = [[1,2]];

% Transport direction in graph
b = [2];

% Define inflow and outflow nodes
InflowNodes = [1];
OutflowNodes = [2];
FreeNodes = setdiff(unique(Edges), union(InflowNodes, OutflowNodes));

% Model parameters
eps = 5.e-2;

% Discretization parameters
ne = [500];
he = 1/ne;
nt = 1000;
tau = 1/nt;

% Boundary and initial conditions
g = @(t)((t<0.2)+(t>0.4)*(t<0.6)+(t>0.8));
u = zeros(sum(ne),nt);
u(1,1) = 1;

% Mass matrix
M = spdiags(ones(ne(1),1)*he(1), 0, ne, ne);

% Advection term
A = spdiags([-max(0,b(1))*ones(ne(1),1), abs(b(1))*ones(ne(1),1), min(0,b(1))*ones(ne(1),1)], [-1,0,1], ne(1), ne(1));

% Correction at inflow and outflow
A(1,1) = max(0,b);
A(ne(1),ne(1)) = b-min(0,b);

% Diffusion term
D = 1/he*spdiags([-ones(ne(1),1), 2*ones(ne(1),1), -ones(ne(1),1)], [-1,0,1], ne(1), ne(1));
D(1,1) = 1/he;
D(ne(1),ne(1)) = 1/he;

% Right-hand side and inflow bc
rhs = zeros(ne(1),1); 

% Solve equation system in stationary case
for k=2:nt
    fprintf("Timestep %d/%d\n", k, nt);
    
    % Assemble right-hand side
    t = k*tau;
    rhs(1) = b(1)*g(t);
    
    % Compute solution at new time point
    u(:,k) = (M+tau*(eps*D+A))\(M*u(:,k-1) + tau*rhs);
    
    % Plot current timestep
    plot(u(:,k));
    ylim([0,2]);
    pause(0.01)
end