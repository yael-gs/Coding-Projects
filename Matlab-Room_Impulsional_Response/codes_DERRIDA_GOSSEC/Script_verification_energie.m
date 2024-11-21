clear all
close all

%% Source

% Paramètres des rayons 
src = [1,1]; % Coordonnées de la source
N_ray = 10000; % Nombre de rayons 
ammortissement_mur = 1;  
%valeur comprise entre 0 et 1 par lequelle l'intensité est multipliée
%lorsque le rayon rebondit sur un mur (voir fct "rebond")
%dès qu'on baisse ce facteur sous 0.75 , l'effet d'écho est largement réduit
ammortissement_micro = 1; %idem pour le micro  

ray = rayons(N_ray,src, ammortissement_mur, ammortissement_micro);

%% Micro

xm = 4;  % Coordonnée X du micro
ym = 4;  % Coordonnée Y du micro
rm = 1; % Rayon de sensibilité du micro

mic = micro(xm,ym,rm);

%%  Calcul et Affichage vérification énergie

hold on;

verif_energie(ray,mic);

hold off;
