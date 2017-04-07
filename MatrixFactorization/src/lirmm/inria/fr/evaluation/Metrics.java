/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lirmm.inria.fr.evaluation;

import lirmm.inria.fr.data.DataMatrix;
import lirmm.inria.fr.math.BigSparseRealMatrix;
import lirmm.inria.fr.math.OpenLongToDoubleHashMap;
import org.apache.commons.math3.util.FastMath;

/**
 *
 * @author rbouadjenek
 */
public class Metrics {

    /**
     * This function aims to optimize the dot product
     */
    private static double getDotProduct(BigSparseRealMatrix U, int col1, BigSparseRealMatrix V, int col2) {
        double dot = 0;
        for (int i = 0; i < U.getRowDimension(); i++) {
            dot += U.getEntry(i, col1) * V.getEntry(i, col2);
        }
        return dot;
    }

    public static void evaluate(DataMatrix R, BigSparseRealMatrix U, BigSparseRealMatrix V) {
        OpenLongToDoubleHashMap rowsMAE = new OpenLongToDoubleHashMap(0.0);
        OpenLongToDoubleHashMap rowsRMSE = new OpenLongToDoubleHashMap(0.0);
        OpenLongToDoubleHashMap rowsT = new OpenLongToDoubleHashMap(0.0);
        //-------------------------------------------------------------------
        OpenLongToDoubleHashMap columnsMAE = new OpenLongToDoubleHashMap(0.0);
        OpenLongToDoubleHashMap columnsRMSE = new OpenLongToDoubleHashMap(0.0);
        OpenLongToDoubleHashMap columnsT = new OpenLongToDoubleHashMap(0.0);

        double RMSE = 0;
        double MAE = 0;
        double RMSERand = 0;
        double MAERand = 0;
        int k = 0;
        for (OpenLongToDoubleHashMap.Iterator iterator = R.getTestSetEntries().iterator(); iterator.hasNext();) {
            k++;
            iterator.advance();
            final double value = iterator.value();
            final long key = iterator.key();
            final int i, j;
            i = (int) (key / R.getColumnDimension());
            j = (int) (key % R.getColumnDimension());
//            double estimation = U.transpose().getRowVector(j).dotProduct(V.getColumnVector(j));
            double estimation = getDotProduct(U, i, V, j);
            RMSE += FastMath.pow(value - estimation, 2);
            MAE += FastMath.abs(value - estimation);
            double r = Math.random() * R.getMax();
            RMSERand += FastMath.pow(value - r, 2);
            MAERand += FastMath.abs(value - r);

            double v = rowsMAE.get(R.getRowNonZeroEntry(i));
            rowsMAE.put(R.getRowNonZeroEntry(i), v + FastMath.abs(value - estimation));
            v = rowsRMSE.get(R.getRowNonZeroEntry(i));
            rowsRMSE.put(R.getRowNonZeroEntry(i), v + FastMath.pow(value - estimation, 2));
            v = rowsT.get(R.getRowNonZeroEntry(i));
            rowsT.put(R.getRowNonZeroEntry(i), v + 1);

            v = columnsMAE.get(R.getColumnNonZeroEntry(j));
            columnsMAE.put(R.getColumnNonZeroEntry(j), v + FastMath.abs(value - estimation));
            v = columnsRMSE.get(R.getColumnNonZeroEntry(j));
            columnsRMSE.put(R.getColumnNonZeroEntry(j), v + FastMath.pow(value - estimation, 2));
            v = columnsT.get(R.getColumnNonZeroEntry(j));
            columnsT.put(R.getColumnNonZeroEntry(j), v + 1);
        }
        System.out.println("******************************************");
        for (OpenLongToDoubleHashMap.Iterator iterator = rowsMAE.iterator(); iterator.hasNext();) {
            iterator.advance();
            final long key = iterator.key();
            double localMAE = rowsMAE.get(key) / rowsT.get(key);
            double localRMSE = FastMath.sqrt(rowsRMSE.get(key) / rowsT.get(key));
            System.out.println(key + "\t" + (int) rowsT.get(key) + "\t" + localMAE + "\t" + localRMSE);
        }
        System.out.println("******************************************");
        for (OpenLongToDoubleHashMap.Iterator iterator = columnsMAE.iterator(); iterator.hasNext();) {
            iterator.advance();
            final long key = iterator.key();
            double localMAE = columnsMAE.get(key) / columnsT.get(key);
            double localRMSE = FastMath.sqrt(columnsRMSE.get(key) / columnsT.get(key));
            System.out.println(key + "\t" + (int) columnsT.get(key) + "\t" + localMAE + "\t" + localRMSE);
        }
        RMSE /= k;
        MAE /= k;
        RMSE = FastMath.sqrt(RMSE);
        RMSERand /= k;
        MAERand /= k;
        RMSERand = FastMath.sqrt(RMSERand);
        System.out.println("******************************************");
        System.out.println("Type\tMAE\tRMSE");
        System.out.println("DPMF\t" + MAE + "\t" + RMSE);
        System.out.println("Random\t" + MAERand + "\t" + RMSERand);
        System.out.println("******************************************");
    }
}
