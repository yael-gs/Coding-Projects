function [rir] = mesure(ancien_ray,nouveau_ray,mic,rir) 

%Cette fonction va mettre a jour la reponse impulsionelle, en ajoutant le nombre de rayons qui intersectent le micro a l'instant pertinent de l'echelle des temps.

%On a besoin de l'ancien_ray en variable de la fonction car notre fonction
%determine quels rayons ont traversés le micro lors de la dernière itération (et non quels rayons VONT traverser le micro lors de l'itération suivante).  

for k=1:ancien_ray.nb
    sol = intersection_mesure(ancien_ray,mic,k);
    sol_1 = sol(1,1);
    sol_2 = sol(1,2);
    if (sol_1>0 & sol_2>0 & nouveau_ray.last_dist(k,1)>max(sol_1,sol_2) ) %si le rayon à effectivement traversé le micro
        D = ancien_ray.dist(k,1) + (sol_1 + sol_2)/2; 
        fs = rir.freq; %la frequence d'echantilonnage doit etre dans rir
        c = rir.celerite; %vitesse du son dans rir
        temps_mis_par_rayon = floor(D*fs/c) +1 ; 
        %{ +1 car les tableaux en matlab sont indicées à partir de 1, NB : ce n'est pas vraiment un temps c'est un indice sans unité %}
        if temps_mis_par_rayon < rir.taille %%faut pas que ca depasse la taille de rir 
        rir.ordonnee(temps_mis_par_rayon) = rir.ordonnee(temps_mis_par_rayon) + ancien_ray.intensity(k); %on incremente la reponse selon l'intensité du rayon reçu
        nouveau_ray.intensity(k) = ancien_ray.intensity(k)*ancien_ray.ammortissement_micro; %le rayon est attenué d'un facteur ammortissement
    end
end
end
end



        
    