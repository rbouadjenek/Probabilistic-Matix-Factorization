/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package lirmm.inria.fr.recommendation.pmf;

import lirmm.inria.fr.math.BigSparseRealMatrix;

/**
 * Structure to represent a point. A point represents the two matrices U and V
 * at a given iteration of the gradient Algorithm.
 *
 * @author rbouadjenek
 */
public class Point {

    private final int latentDimension; // number of latent dimension
    private BigSparseRealMatrix U; // The user's latent features
    private BigSparseRealMatrix V; // The item's latent features
    private double J; // Value of the cost function J

    /**
     * Initializes a newly created Point object so that it represents U and V
     * matrices.
     *
     * @param numUsers The number of Users
     * @param numItems The number of Items
     * @param latentDimension The number of latent dimensions
     */
    public Point(int numUsers, int numItems, int latentDimension) {
        this.latentDimension = latentDimension;
        if (numUsers == 0 || numItems == 0) {
            return;
        }
        U = new BigSparseRealMatrix(latentDimension, numUsers);
        V = new BigSparseRealMatrix(latentDimension, numItems);
        J = Double.MAX_VALUE;
    }

    /**
     * Copy this point to the point given in parameter.
     *
     * @param to a point to be copied
     */
    public void copy(Point to) {
        to.U = U.copy();
        //---------------------------
        to.V = V.copy();
        to.J = J;
    }

    /**
     * This method returns the matrix containing the User features.
     *
     * @return The matrix containing the User features.
     */
    public BigSparseRealMatrix getU() {
        return U;
    }

    /**
     * This method returns the matrix containing the Item features.
     *
     * @return The matrix containing the Item features.
     */
    public BigSparseRealMatrix getV() {
        return V;
    }

    /**
     * This method returns the value of the cost function of this point
     * (Evaluation of the point).
     *
     * @return The value of the cost function of this point (Evaluation of the
     * point).
     */
    public double getJ() {
        return J;
    }

    /**
     * This method update the value of the cost function of this point
     * (Evaluation of the point).
     *
     * @param J set the value of J
     */
    public void setJ(double J) {
        this.J = J;
    }

    /**
     * This method initialize the current point (the latent matrices U and V)
     * with smal random values.
     */
    public void initialize() {
        U = BigSparseRealMatrix.randomGenerateMatrix(latentDimension, U.getColumnDimension());
        U = U.scalarMultiply(0.1);
        V = BigSparseRealMatrix.randomGenerateMatrix(latentDimension, V.getColumnDimension());
        V = V.scalarMultiply(0.1);

//        System.out.println("U=");
//        System.out.println(U);
//        System.out.println("V=");
//        System.out.println(V);
        double[][] uData = {{0.04459, 0.06017, 0.09179, 0.04057, 0.08069, 0.06254, 0.07317, 0.02529, 0.0926, 0.04346},
        {0.06973, 0.049, 0.06391, 0.01254, 0.05567, 0.05333, 0.08618, 0.05368, 0.01709, 0.06473},
        {0.08956, 0.09606, 0.06459, 0.09891, 0.04529, 0.04609, 0.08478, 0.05972, 0.01564, 0.06002}};

        double[][] vData = {{0.06608, 0.02523, 0.0624, 0.0897, 0.03538, 0.05164, 0.06457, 0.07063, 0.09929, 0.08286},
        {0.05791, 0.09987, 0.05437, 0.03309, 0.04333, 0.02943, 0.01195, 0.01118, 0.03416, 0.01022},
        {0.00973, 0.08802, 0.05382, 0.06729, 0.07961, 0.03695, 0.02192, 0.06149, 0.00978, 0.04965}};
        U = new BigSparseRealMatrix(uData);
        V = new BigSparseRealMatrix(vData);
    }

    public void setU(BigSparseRealMatrix U) {
        this.U = U;
    }

    public void setV(BigSparseRealMatrix V) {
        this.V = V;
    }

}
