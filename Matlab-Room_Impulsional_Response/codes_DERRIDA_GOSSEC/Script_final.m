
%% Nettoyage

clear all
close all



%% Temps
temps_global = tic; % Tic pour mesurer la durée d'execution totale

%% Users

% Paramètres de la salle
vtx = [0,0;0,2;2 2; 2 4; 0 4; 0 6; 6 6; 6 0];
elt = [1,2;2,3;3,4;4,5;5,6;6,7;7,8;8,1];

% Paramètres des rayons 
src = [1,1]; % Coordonnées de la source
N_ray = 1000; % Nombre de rayons 
ammortissement_mur = 0.1;  
%valeur comprise entre 0 et 1 par lequelle l'intensité est multipliée
%lorsque le rayon rebondit sur un mur (voir fct "rebond")
%dès qu'on baisse ce facteur sous 0.75 , l'effet d'écho est largement réduit
ammortissement_micro = 0.1; %idem pour le micro 

% Micro
xm = 3;  % Coordonnée X du micro
ym = 2;  % Coordonnée Y du micro
rm = 0.2; % Rayon du micro

% Paramètres d'enregistrement 
fs = 44100; %freq echantillonage
duree = 100;
n_iter = 150; %si on met plus de 158 itérations (par exemple 175), le code bug independamment du nombre de rayons (la fonction rebond affiche une erreur qui n'apparait pas a moins de 150 itérations... on n'a pas su debugger cela)
celerite = 340; %célérité du son utilisée


%% Géométrie

temps_init = tic; %tic pour la durée d'initialisation

mesh = geometry(vtx,elt); %intialisation du maillage
ray = rayons(N_ray,src,ammortissement_mur,ammortissement_micro); %initialisation des rayons
mic = micro(xm,ym,rm); %intialisation du micro
fprintf("durée de l'initialisation: ");
toc(temps_init);

%% Affichage 
temps_affichage = tic;
figure(1)

plotmaillage(mesh);
plotray(ray);
plotmicro(mic);

fprintf("durée de l'affichage de la salle, micro et rayons initiaux: ");
toc(temps_affichage);


%% Création et mise à jour de la Réponse Impulsionnelle de la Salle
temps_rir = tic;

rir = initialisation_rir(fs,duree,celerite); %initialise l'objet RiR

for i=1:n_iter  % Boucle qui met à jour le tableau rir
        
        new_ray = rebond(ray,mesh);
        rir = mesure(ray,new_ray,mic,rir);
        ray = new_ray;
end
plotray(ray);

rir.ordonnee = rir.ordonnee / max(rir.ordonnee); %normalisation du vecteur rir

fprintf("durée des itérations : ");
toc(temps_rir);

%% Affichage de la Réponse Impulsionnelle de la Salle
figure(2)
temps_affichage_rir = tic;
stem(rir.ordonnee,'.'); 
fprintf("durée de l'affichage de la RiR ");
toc(temps_affichage_rir);

fprintf("durée totale du code ");
toc(temps_global);

%% Lecture fichier audio
[audio,fs] = audioread('zipette.mp3');
y = rir.ordonnee;
sortie = fftfilt(audio,y);
sortie = sortie/(max(sortie));
sound(sortie,fs);