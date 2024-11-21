%% Nettoyage

clear all
close all

%% Users
agg= 1; %en aggrandissant ce parametre les ondes arrivent aux écarts voulus

% Salle
vtx = [0 0; 4 4; 0 4; 4 0]*agg;
elt = [2 4; 1 3; 3 2; 4 1];

% Source
src = [1,2]*agg; % Coordonnées source
N_ray = 1000; % Nombre de rayons 
ammortissement_mur = 1;  
%valeur comprise entre 0 et 1 par lequelle l'intensité est multipliée
%lorsque le rayon rebondit sur un mur (voir fct "rebond")
ammortissement_micro = 1; %idem pour le micro 


% Micro
xm = 3*agg;  % Coordonnée X du micro
ym = 2*agg;  % Coordonnée Y du micro
rm = 0.1; % Rayon de sensibilité du micro

% Paramètres d'enregistrement 
fs = 44100; %freq echantillonage
duree = 1*10e-2*agg; %on multiplie par agg car si la salle est petite les rayons arrivent plus vite donc pas la peine d'avoir un grand tableau de RiR
n_iter = 2; 
celerite = 340; %célérité du son utilisée

%% Initialisation

mesh = geometry(vtx,elt); 
mic = micro(xm,ym,rm);
ray = rayons(N_ray,src, ammortissement_mur, ammortissement_micro);
    
%% Affichage 
figure(1);
plotmaillage(mesh);
plotray(ray);
plotmicro(mic);


%% Création et mise à jour de la Réponse Impulsionnelle de la Salle

rir = initialisation_rir(fs,duree,celerite); %initialise l'objet RiR

for i=1:n_iter  % Boucle qui met à jour le tableau rir
        
        new_ray = rebond(ray,mesh);
        rir = mesure(ray,new_ray,mic,rir);
        plot_lignes(ray,new_ray); %(pour afficher les rebonds) 
        ray = new_ray;
end

rir.ordonnee = rir.ordonnee / max(rir.ordonnee); %normalisation du vecteur

%% Affichage de la Réponse Impulsionnelle de la Salle
figure(2);
stem(rir.ordonnee,'.'); 

