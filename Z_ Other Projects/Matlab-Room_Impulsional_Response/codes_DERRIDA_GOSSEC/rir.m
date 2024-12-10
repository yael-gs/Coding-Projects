function rir = initialisation_rir(fs, duree, celerite)

%cette fonction initialise l'objet rir

rir.ordonnee = zeros(duree*fs);
rir.taille = duree*fs;
rir.freq = fs;
rir.celerite = celerite;

end

