/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package lirmm.inria.fr.recommendation.pmf;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;
import lirmm.inria.fr.data.DataMatrix;
import lirmm.inria.fr.evaluation.Metrics;
import lirmm.inria.fr.main.Functions;
import lirmm.inria.fr.math.BigSparseRealMatrix;

/**
 * This is the main class that solves a problem of matrix
 * factorization using the gradient algorithm. For a given problem, this class
 * would lunch the gradient algorithm.
 *
 * @author rbouadjenek
 */
public class Solver {

    private final boolean debug;
    private GradientDescent problem;

    /**
     * Create a structure that aims at solve a matrix factorization problem.
     *
     * @param r A matrix to factorize.
     * @param latentDimension Number of latent dimensions.
     * @param numEvaluations Maximum number of iterations in the gradient
     * algorithm.
     * @param lambda Weight of the social regularization terms.
     * @param debug debug or not?
     */
    public Solver(BigSparseRealMatrix r, int latentDimension, int numEvaluations, double lambda, boolean debug) {
        this.debug = debug;
        try {
            problem = new GradientDescent(r, latentDimension, numEvaluations, lambda);
        } catch (Exception ex) {
            Logger.getLogger(Solver.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * This method solve a factorization problem. The latent items and users
     * features and availble in problem.getNextPoint(). We can predict missing
     * values in the factorized matrix by multiplying U x V
     */
    public void solve() {
        problem.findMinimum();

        // Cette instruction bouffe du temps de calcule,
        // pas la peine de l'éxécuter en entié, mais multiplier uniquement la ligne par la colonne qui m'interesse....
        System.err.println("Best point= " + problem.getCurrentPoint().getJ());
        System.err.println("Evaluations = " + problem.getNumEvaluations());
        if (debug) {
            BigSparseRealMatrix prod = this.problem.getCurrentPoint().getU().transpose().multiply(this.problem.getCurrentPoint().getV());
//            System.out.println("Best point= " + problem.getNextPoint().getJ());
//            System.out.println("Evaluations = " + problem.getNumEvaluations());
            System.out.println();
            System.out.println("U:");
            System.out.println(problem.getCurrentPoint().getU());
            System.out.println("V:");
            System.out.println(problem.getCurrentPoint().getV());
            System.out.println("Prod:");
            System.out.println(prod);
            System.out.println("R:");
            System.out.println(problem.getR());
            printExistingValues();
        }
    }

    /**
     *
     */
    public void printExistingValues() {
        DecimalFormat df;
        df = new DecimalFormat();
        df.setMaximumFractionDigits(4); //arrondi à 2 chiffres apres la virgules
        df.setMinimumFractionDigits(2);
        df.setDecimalSeparatorAlwaysShown(true);
        System.out.println("Existing values are: ");
        double z = 0;
        for (int i = 0; i < problem.getR().getRowDimension(); i++) {
            for (int j = 0; j < problem.getR().getColumnDimension(); j++) {
                if (problem.getR().getEntry(i, j) != 0) {
                    System.out.println(df.format(problem.getR().getEntry(i, j)) + " <-> " + df.format(this.problem.getCurrentPoint().getU().transpose().getRowVector(i).dotProduct(this.problem.getCurrentPoint().getV().getColumnVector(j))));
//                    z += FastMath.pow(problem.getR().getEntry(i, j) - this.problem.getCurrentPoint().getU().transpose().getRowVector(i).dotProduct(this.problem.getCurrentPoint().getV().getColumnVector(j)), 2);
                }
            }
        }
//        System.out.println("z= "+z/2);
    }

    public void printRMSE() {
        Metrics.evaluate((DataMatrix) problem.getR(), this.problem.getCurrentPoint().getU(), this.problem.getCurrentPoint().getV());
//        System.out.println("RMSE= " + RMSE);
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here

        long start = System.currentTimeMillis();
//        double[][] sData = {{0.84, 0.00, 0.00, 0.96, 0.39,}, {0.00, 0.00, 0.00, 0.49, 0.00,}, {0.00, 0.00, 0.00, 0.00, 0.00,}, {0.41, 0.65, 0.12, 0.00, 0.15,}, {0.43, 0.00, 0.00, 0.63, 0.29,}};
//
        double[][] rData = {{0.00000, 0.00000, 0.55981, 0.00000, 0.00000, 0.00000, 0.00000, 0.72600, 0.00000, 0.26447},
        {0.00000, 0.58614, 0.35670, 0.00000, 0.91853, 0.00000, 0.77274, 0.00000, 0.00000, 0.00000},
        {0.25913, 0.00000, 0.15131, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.84046},
        {0.42690, 0.00000, 0.03426, 0.62947, 0.64010, 0.80451, 0.00000, 0.00000, 0.00000, 0.85114},
        {0.00000, 0.00000, 0.00000, 0.26811, 0.00000, 0.00000, 0.21317, 0.03503, 0.00000, 0.00000},
        {0.00000, 0.36122, 0.66181, 0.15107, 0.25372, 0.00000, 0.47690, 0.16886, 0.00000, 0.99222},
        {0.00000, 0.00000, 0.00000, 0.00000, 0.88366, 0.24212, 0.00000, 0.00000, 0.00000, 0.00000},
        {0.00000, 0.00000, 0.00000, 0.26811, 0.00000, 0.00000, 0.21317, 0.03503, 0.00000, 0.00000},
        {0.00000, 0.36122, 0.66181, 0.15107, 0.25372, 0.00000, 0.47690, 0.16886, 0.00000, 0.99222},
        {0.00000, 0.00000, 0.00000, 0.00000, 0.88366, 0.24212, 0.00000, 0.00000, 0.00000, 0.00000}};

        BigSparseRealMatrix r = new BigSparseRealMatrix(rData);
//        DataMatrix r = DataMatrix.createDataMatrix("/Volumes/Macintosh HD/Users/rbouadjenek/Documents/PostDoc/datasets/testData/ratingsTest.txt");
//        r.cutDataSet(10, 0);
//        r = new DataMatrix("/Volumes/Macintosh HD/Users/rbouadjenek/Documents/PostDoc/datasets"
//                + "/foursquare/myfoursquare_dataset/ratings.txt");
        r = BigSparseRealMatrix.randomGenerateMatrix( 1000000, 1000000, 0.000001);
//        r = BigSparseRealMatrix.randomGenerateMatrix(7, 10, 40);
//        r = BigSparseRealMatrix.randomGenerateMatrix(5573, 100, 3);
//        System.out.println(r);
//        double sparsity = (double) r.getDataSize() * 100 / ((double) r.getRowDimension() * r.getColumnDimension());
//        System.out.println("R=[" + r.getRowDimension() + "," + r.getColumnDimension() + "]= " + r.getDataSize() + " (sparsity= " + sparsity + "%)");
//        System.out.println(r);
        Solver s = new Solver(r, 2, 120, 0.02, true);
        s.solve();
        long end = System.currentTimeMillis();
        System.err.println("Global Execution time: " + Functions.getTimer(end - start) + ".");
//        s.printExistingValues();
//        s.printRMSE();

    }
}
