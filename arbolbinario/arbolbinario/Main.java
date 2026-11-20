package arbolbinario;

import java.util.Iterator;

public class Main {
    public static void main(String[] args) {
        ArbolBinario<String> subIzquierdo = new ArbolBinario<>("B", null, null);
        ArbolBinario<String> subDerecho = new ArbolBinario<>("C", null, null);
        ArbolBinario<String> arbol = new ArbolBinario<>("A", subIzquierdo, subDerecho);

        System.out.println("Tamaño del árbol: " + arbol.size());
        System.out.println("¿Contiene 'B'?: " + arbol.contains("B"));

        System.out.print("Recorrido PreOrden: ");
        Iterator<String> it = arbol.iteratorPreOrden();
        while (it.hasNext()) {
            System.out.print(it.next() + " ");
        }
        System.out.println();
    }
}