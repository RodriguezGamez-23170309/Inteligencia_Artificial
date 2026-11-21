/* Omar Ernesto Rodriguez Gamez  - 23170309
* Inteligencia Artificial 6:00 - 7:00 p.m.
* 8 Puzzle   */
package arbol;

public class Node implements Comparable<Node> {
    private String state;
    private Node parent;
    private int depth;
    private int cost;

    public Node() {
    }

    public Node(String state, Node parent, int depth, int cost) {
        this.state = state;
        this.parent = parent;
        this.depth = depth;
        this.cost = cost;
    }

    public Node(String state, Node parent) {
        this.state = state;
        this.parent = parent;
        this.depth = (parent == null) ? 0 : parent.getDepth() + 1;
        this.cost = 0;
    }

    public String getState() { return state; }
    public void setState(String state) { this.state = state; }

    public Node getParent() { return parent; }
    public void setParent(Node parent) { this.parent = parent; }

    public int getDepth() { return depth; }
    public void setDepth(int depth) { this.depth = depth; }

    public int getCost() { return cost; }
    public void setCost(int cost) { this.cost = cost; }

    @Override
    public int compareTo(Node other) {
        return Integer.compare(this.depth, other.depth);
    }
}