function [solutions] = intersection_mesure(ray,mic,k) 

% cette fonction est une variante de la fonction intersection, on l'utilise dans la fonction mesure. 
% Elle renvoie les deux solutions (si elles existent) d'intersection entre le rayon en position k et le micro
    
    sol_1 = -10;
    sol_2 = -10; %par defaut ainsi on rentrera pas dans le if de la fonction mesure si il ny a pas de solutions
    xm = mic.pos(1,1);
    ym = mic.pos(1,2);
    a = ray.pos(k,1);
    b = ray.pos(k,2);
    ux = ray.dir(k,1);
    uy = ray.dir(k,2);
    B = 2*uy*(b-ym)+ 2*ux*(a-xm);
    D = (a-xm)^2+ (b-ym)^2 - mic.rad^2;
    det = B^2 - 4*D; % c'est l'expression du determinant (calcul de l'expression fait au brouillon) 
    
    if det>=0
            sol_1 = (-B - sqrt(det)) / 2;
            sol_2 = (-B + sqrt(det))/ 2;
            
    end
    solutions(1,1) = sol_1;
    solutions(1,2) = sol_2;
    end
    
            
            


