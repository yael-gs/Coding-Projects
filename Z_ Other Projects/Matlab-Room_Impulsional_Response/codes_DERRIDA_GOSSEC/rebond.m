function [ray] = rebond(ray,mesh) 


%{ Cette fonction met a jour le tableau des rayons, avec leur nouveau point source (point d'intersection avec le mur) et leur nouveau vecteur directeur et leur nouvelle intensité. %};

%pour trouver le point d'intersection, on résoud matriciellement (fait sur papier)

for k=1:size(ray.dir,1)
    lambda_min = Inf ; %on réinitialise le lambda_min a l'infini pour chaque rayon
    indice_intersect = -1; %ligne du mur que le rayon k va intersecter (on initialise à une valeur quelquonque au debut sauf entiers naturels)
    indice_teste = -1; %l'interet de cet indice et de ne pas faire les memes calculs plusieurs fois (val quelquonque aussi)
    ux = ray.dir(k,1); %cf notations  de la feuille
    uy = ray.dir(k,2);
    xs = ray.pos(k,1);
    ys = ray.pos(k,2);
    for i=1:size(mesh.elt,1)  %on va tester tous les murs 
        tx = mesh.tgt(i,1);
        ty = mesh.tgt(i,2);
        xm = mesh.ctr(i,1); %xm pour xmur
        ym = mesh.ctr(i,2);
        A = [ux, -tx; uy,-ty];
        B = [xm-xs; ym-ys];
        if abs(det(A))>10e-3     %NB rigoureusement c'est just det(A) != 0 mais on rajoute une marge , cette condition verifie que les droites ne sont pas parallèles
                 X = linsolve(A,B);
                 a = X(1,1);
                 b = X(2,1);
                 if abs(b)<=mesh.lgt(i,1)/2 %%verif que on se trouve bien sur le mur
                     if a>0 %%verif qu'on sest propagé dans le bon sens 
                         if a<=lambda_min
                             indice_intersect = i; %on modifie la ligne d'intersect ssi toutes les conditions sont verifiées
                         end
                     end
                 end
        end
        %pour trouver le nouveau vecteur directeur
        
        %Rq: on peut optimiser ça en le sortant dela boucle mais ça bug
        
        if indice_intersect ~= indice_teste  %permet d'éviter de refaire les calculs a chaque boucle, on le fait juste si l'indice à changé
            indice_teste = indice_intersect;
            a = a -1.0e-5; %%pour eviter les pb aux niveaux des coins
            lambda_min = a;
            Tx = mesh.tgt(indice_intersect,1);
            Ty = mesh.tgt(indice_intersect,2);
            Nx = mesh.nrm(indice_intersect,1);
            Ny = mesh.nrm(indice_intersect,2);
            u_prime_x =(ux*Tx+uy*Ty)*Tx -(ux*Nx+uy*Ny)*Nx;
            u_prime_y =(ux*Tx+uy*Ty)*Ty -(ux*Nx+uy*Ny)*Ny;
            indicateur = 1;                
        end
    end
    
    ray.dir(k,1) = u_prime_x;
    ray.dir(k,2) = u_prime_y; 
    ray.pos(k,1) = xs +lambda_min*ux;
    ray.pos(k,2) = ys + lambda_min*uy;
    dist = lambda_min;
    ray.dist(k,1) = ray.dist(k,1) + dist ; %on garde en mémoire la distance parcourue par chaque rayon
    ray.last_dist(k,1) = dist; %on garde en mémoire la dernière distance 
    ray.intensity(k) = ray.intensity(k)*ray.ammortissement_mur; %on amortit le rayon 
end
end
    
        
    

