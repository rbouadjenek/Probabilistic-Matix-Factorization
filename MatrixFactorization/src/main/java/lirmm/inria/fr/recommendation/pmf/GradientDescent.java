/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package lirmm.inria.fr.recommendation.pmf;

import java.text.DecimalFormat;
import lirmm.inria.fr.math.BigSparseRealMatrix;
import org.apache.commons.math3.util.FastMath;

/**
 * This class represents the gradient descent algorithm. This class includes all
 * methods for computing the gradient features as well as the objective
 * function.
 *
 * @author rbouadjenek
 */
public final class GradientDescent {

    private final int maxNumEvaluations;// maximum number of iteration for the gradient descent algorithm
    final double lambda; // Lambda for the regularization terms
    final int latentDimension; // Number of latent dimension
    private Point nextPoint;
    private Point currentPoint;
    private BigSparseRealMatrix R;
    private int numEvaluations;
    private final double epsilone = 0.000005; // Stop creteria for the gradient descent algorithm
    /**
     * Step size parameters
     */
    protected double currentStepSize = 0.1;//The current step size.
    protected double increasedCostPercentage = 0.10; //the percentage of the learning rate that will be used when cost increases.
    protected double decreasedCostPercentage = 0.7; //the percentage of the learning rate that will be used when cost decreases.
    protected double minStepSize = 0.00000000001;// Minimum value for the step size

    /**
     * ********************************************
     *
     */
    /**
     * Create a structure that represent the problem of matrix factorization.
     *
     * @param R The matrix to factorize.
     * @param latentDimension Number of latent dimensions used in the
     * factorization process.
     * @param numEvaluations Number of maximum evaluation.
     * @param lambda Weight for the regularization terms.
     * @throws Exception
     */
    public GradientDescent(BigSparseRealMatrix R, int latentDimension, int numEvaluations, double lambda) throws Exception {
        this.maxNumEvaluations = numEvaluations;
        this.numEvaluations = 0;
        if (R.getRowDimension() == 0 || R.getColumnDimension() == 0) {
            throw new Exception("Check the rows and the columns of the matrix R.");
        }
        this.nextPoint = new Point(R.getRowDimension(), R.getColumnDimension(), latentDimension);
        this.nextPoint.initialize();
        this.currentPoint = new Point(R.getRowDimension(), R.getColumnDimension(), latentDimension);
        nextPoint.copy(currentPoint);
        this.lambda = lambda;
        this.latentDimension = latentDimension;
        this.R = R;
    }

    /**
     * This function is the objective function. It is called by Gradient class
     * to evaluate current point.
     */
    private double computeCostFunction(BigSparseRealMatrix U, BigSparseRealMatrix V) {
        double J = ((double) 1 / 2) * FastMath.pow(((R.specialOperation(U.transpose(), V)).getFrobeniusNorm()), 2); // compute (R-U'*V).^2
        J += (lambda / 2) * FastMath.pow(U.getFrobeniusNorm(), 2);// Adding the regularization term for U
        J += (lambda / 2) * FastMath.pow(V.getFrobeniusNorm(), 2);// Adding the regularization term for V
        return J;
    }

    /**
     * This function is the derivative of objective function. It is called by
     * the Gradient class to evaluate and update the current point. This method
     * should populate the elements of the array point.gradient.
     */
    private void computeGradient() {
        BigSparseRealMatrix temp = R.specialOperation(this.nextPoint.getU().transpose(), this.nextPoint.getV()).scalarMultiply(-1);// compute (U'*V-R)
        //--------------------------------------------------------------
        //------------ Compute the new values of U------------------------
        //--------------------------------------------------------------
        BigSparseRealMatrix updatedU = temp.multiply(this.nextPoint.getV().transpose());// compute (U'*V-R)*V'
//        System.out.println(updatedU);
        updatedU = updatedU.add(this.nextPoint.getU().transpose().scalarMultiply(lambda));// compute (U'*V-R)*V'+lambda*U'
        updatedU = this.nextPoint.getU().subtract(updatedU.transpose().scalarMultiply(currentStepSize));// compute U=U-alpha*J'
        //--------------------------------------------------------------
        //------------ Compute the new values of V ----------------------
        //--------------------------------------------------------------
        BigSparseRealMatrix updatedV = temp.transpose().multiply(this.nextPoint.getU().transpose());// compute (U'*V-R)*U'
//        System.out.println(updatedV);
        updatedV = updatedV.add(this.nextPoint.getV().transpose().scalarMultiply(lambda));// compute (U'*V-R)*U'+lambda*V'
        updatedV = this.nextPoint.getV().subtract(updatedV.transpose().scalarMultiply(currentStepSize));// compute V=V-alpha*J'
        //--------------------------------------------
        //------------ Simultanuously update U and V -------
        //------------------------------------------------
//        System.out.println("---------------");
//        System.out.println(updatedU);
//        System.out.println(updatedV);
        this.nextPoint.setU(updatedU);
        this.nextPoint.setV(updatedV);
    }

    /**
     * This function is will execute an iteration of the gradient algorithm. *
     */
    public void nextIteration() {
        computeGradient();
        double J = computeCostFunction(this.nextPoint.getU(), this.nextPoint.getV());
        this.nextPoint.setJ(J);
        numEvaluations++;
    }

    /**
     * This method return the current point of the Algorithm.
     *
     * @return The current point of the Algorithm.
     */
    public Point getNextPoint() {
        return nextPoint;
    }

    /**
     * This method return the next point of the Algorithm.
     *
     * @return The current point of the Algorithm.
     */
    public Point getCurrentPoint() {
        return currentPoint;
    }

    /**
     * This is the main matrix to factorize.
     *
     * @return The current matrix to factorize.
     */
    public BigSparseRealMatrix getR() {
        return R;
    }

    /**
     * The number of the current iteration of the algorithm.
     *
     * @return The current iteration of the algorithm.
     */
    public int getNumEvaluations() {
        return numEvaluations;
    }

    /**
     * This is the main optimisation routine that uses gradient descent. This
     * class will also change the learning rate over time by observing the cost
     * of the cost function. If the cost decreases, it will increase the learning
     * rate (typically by 5%). If the cost increases it will (typically) cut the
     * learning rate in half.
     */
    public void findMinimum() {
        DecimalFormat df;
        df = new DecimalFormat();
        df.setMaximumFractionDigits(4); //arrondi à 2 chiffres apres la virgules
        df.setMinimumFractionDigits(2);
        df.setDecimalSeparatorAlwaysShown(true);
        double J = computeCostFunction(this.nextPoint.getU(), this.nextPoint.getV());
        this.currentPoint.setJ(J);
        System.err.println("*********** Initial J ***********");
        System.err.println("J= " + currentPoint.getJ());
        System.err.println("*********************************");
        while (numEvaluations < maxNumEvaluations) {
            nextIteration();	// Computes the gradient and calculates the next point with the current step size            
            if (nextPoint.getJ() < currentPoint.getJ()) {
                double rate = (currentPoint.getJ() - nextPoint.getJ()) * 100 / currentPoint.getJ();
                nextPoint.copy(currentPoint);
                currentStepSize += currentStepSize * increasedCostPercentage; //increase the learning rate (typically by 5%).
                System.err.println("+++++++++++" + currentStepSize);
                System.err.println("J= " + df.format(nextPoint.getJ()) + " (rate: " + df.format(rate) + "%)");
                if (rate < epsilone) {
                    break;
                }
            } else {
                currentStepSize *= decreasedCostPercentage; // cut the learning rate in half.
                System.err.println("-----------" + currentStepSize);
                System.err.println("Next point= " + df.format(nextPoint.getJ()) + " > current point=" + df.format(currentPoint.getJ()));
                currentPoint.copy(nextPoint);
            }
//            if (currentStepSize < minStepSize) {
//                break;
//            }
            System.err.println("******************" + numEvaluations + "******************");
        }
    }
}
