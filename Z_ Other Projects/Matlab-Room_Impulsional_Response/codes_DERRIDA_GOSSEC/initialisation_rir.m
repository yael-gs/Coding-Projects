function rir = initialisation_rir(fs,duree,celerite) % Cette fonction initialise l'objet rir

rir.celerite = celerite;
rir.ordonnee = zeros(duree*fs,1); 
rir.taille = duree*fs;
rir.freq = fs;


end