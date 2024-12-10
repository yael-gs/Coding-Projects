function plotmaillage(mesh) % Cette fonction affiche la salle avec les centres des murs et les vecteurs normaux et %tangents des centres.

hold on;

p = 0.8; %permet de diminuer la taille des vecteurs sur le dessin


for k =1: size (mesh.vtx,1)
    
    plot(mesh.vtx(k,1),mesh.vtx(k,2),'kx'); % Donnees au noeud
       
end
for k =1:size(mesh.elt,1) 
    
    i = mesh.elt(k,1);
    j = mesh.elt(k,2);
    
        
    plot(mesh.vtx([i,j],1), mesh.vtx([i,j],2),'k-');  % Trace la droite entre deux points
    plot(mesh.ctr(i,1),mesh.ctr(i,2),'xr'); % Trace les centres des murs
    quiver(mesh.ctr(i,1),mesh.ctr(i,2), mesh.nrm(i,1)*p,mesh.nrm(i,2)*p,'g'); % Trace les vecteurs normaux des centres
    quiver(mesh.ctr(i,1),mesh.ctr(i,2), mesh.tgt(i,1)*p,mesh.tgt(i,2)*p,'b'); % Trace les vecteurs tangents des centres
    
end
hold off

axis equal
grid on

end