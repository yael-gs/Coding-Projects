function [mesh] = geometry(noeuds, elements)  %a partir des noeuds et des elements entres par l'user, on crée le maillage et ajoutons des donnees pertinentes dans l'objet Mesh
    mesh.vtx = noeuds ;
    mesh.elt = elements ;
    [n, ~] = size(mesh.elt); %on sait qu'il ya 2 colonnes car on est en 2D, par contre on obtient le nombre n de noeuds
    mesh.ctr = zeros(n,2); 
    mesh.nrm = zeros(n,2);
    mesh.tgt = zeros(n,2);
    mesh.lgt = zeros(n,1);
        for k =1:n %on parcourt le tableau des élements
        i = mesh.elt(k,1); %indice noeud départ
        j = mesh.elt(k,2); %indice noeud arrivée
        x_d = mesh.vtx(i,1) ; % x départ , x final y départ y final 
        x_f = mesh.vtx(j,1);
        y_d = mesh.vtx(i,2) ;
        y_f = mesh.vtx(j,2);
        mesh.ctr(k,1) = (x_d + x_f) / 2 ;% x moyen du k ieme element
        mesh.ctr(k,2) = (y_d + y_f) / 2; % y moyen du k-ieme element 
        norme = sqrt((x_f-x_d)^2+(y_f-y_d)^2); % 
        mesh.lgt(k,1) = norme;
        mesh.tgt(k,1) = (x_f-x_d)/norme; % On définit le vecteur tangents au mur au niveau des centres
        mesh.tgt(k,2) = (y_f - y_d) / norme;
        mesh.nrm(k,1) = mesh.tgt(k,2); % On définit le vecteur normal au mur au niveau des centres
        mesh.nrm(k,2) = -mesh.tgt(k,1);
        end
    end