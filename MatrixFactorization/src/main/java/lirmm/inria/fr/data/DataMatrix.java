/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lirmm.inria.fr.data;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.StringTokenizer;
import lirmm.inria.fr.math.BigSparseRealMatrix;
import lirmm.inria.fr.math.OpenLongToDoubleHashMap;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;

/**
 *
 * @author rbouadjenek
 */
public final class DataMatrix extends BigSparseRealMatrix {

    /**
     * Mapping for rows.
     */
    private final Map<String, Integer> rowsMapping;
    /**
     * Mapping for columns.
     */
    private final Map<String, Integer> columnsMapping;
    /**
     * Mapping for columns.
     */
    private final Map<Integer, Double> columnMeans;

    /**
     * Storage of the test dataset elements.
     */
    private final OpenLongToDoubleHashMap testSetEntries;

    /**
     * Storage of the validation dataset elements.
     */
    private final OpenLongToDoubleHashMap validationSetEntries;

    /**
     * Number of non zero entries in rows of the matrix.
     */
    private final OpenLongToDoubleHashMap rowNonZeroEntries = new OpenLongToDoubleHashMap(0.0);

    /**
     * Number of non zero entries in rows of the matrix.
     */
    private final OpenLongToDoubleHashMap columnNonZeroEntries = new OpenLongToDoubleHashMap(0.0);
    
    /**
     * Maximum value in the matrix
     */
    private double max;
    
    
    private DataMatrix(String file, int rowDimension, int columnDimension, double max) throws NotStrictlyPositiveException, NumberIsTooLargeException {
//        super(5573, 100);
        super(rowDimension, columnDimension);
        this.rowsMapping = new HashMap<>();
        this.columnsMapping = new HashMap<>();
        this.columnMeans = new HashMap<>();
        this.testSetEntries = new OpenLongToDoubleHashMap(0.0);
        this.validationSetEntries = new OpenLongToDoubleHashMap(0.0);
        this.max=max;
        loadMatrix(file, rowDimension, columnDimension);
//        normalize();
    }

    protected void loadMatrix(String file, int rowDimension, int columnDimension) {
        FileInputStream fstream;
        try {
            fstream = new FileInputStream(file);
            // Get the object of DataInputStream
            DataInputStream in = new DataInputStream(fstream);

            Set<Integer> cI = new HashSet();
            for (int i = 0; i < rowDimension; i++) {
                cI.add(i);
            }
            Set<Integer> cJ = new HashSet();
            for (int j = 0; j < columnDimension; j++) {
                cJ.add(j);
            }
            try (BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
                String str;
                int z = 0;
                while ((str = br.readLine()) != null) {
                    str = str.trim();
                    if (str.startsWith("#")) {
                        continue;
                    }
                    if (str.trim().length() == 0) {
                        continue;
                    }
                    z++;
                    StringTokenizer st = new StringTokenizer(str);
                    String row_id = st.nextToken();
                    String column_id = st.nextToken();
                    double rating = Double.parseDouble(st.nextToken());
                    Integer i, j;
                    if (rowsMapping.containsKey(row_id)) {
                        i = rowsMapping.get(row_id);
                    } else {
                        i = cI.iterator().next();
//                        i = cI.get(0);
                        cI.remove(i);
                        rowsMapping.put(row_id, i);
                    }
                    if (columnsMapping.containsKey(column_id)) {
                        j = columnsMapping.get(column_id);
                    } else {
                        j = cJ.iterator().next();
//                        j = cJ.get(0);
                        cJ.remove(j);
                        columnsMapping.put(column_id, j);
                    }
                    setEntry(i, j, rating);
                }
            }

            for (OpenLongToDoubleHashMap.Iterator iterator = getEntries().iterator(); iterator.hasNext();) {
                iterator.advance();
                final long key = iterator.key();
                final int i, j;
                if (isTransposed()) {
                    j = (int) (key / getRowDimension());
                    i = (int) (key % getRowDimension());
                } else {
                    i = (int) (key / getColumnDimension());
                    j = (int) (key % getColumnDimension());
                }
                double v = rowNonZeroEntries.get(i);
                rowNonZeroEntries.put(i, v + 1);
                v = columnNonZeroEntries.get(j);
                columnNonZeroEntries.put(j, v + 1);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void normalize() {
        int[] nbrElements;
        Map<Long, Double> entry = new HashMap();
        if (isTransposed()) {
            nbrElements = new int[getRowDimension()];
        } else {
            nbrElements = new int[getColumnDimension()];
        }
        for (OpenLongToDoubleHashMap.Iterator iterator = getEntries().iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            final long key = iterator.key();
            final int j;
            if (isTransposed()) {
                j = (int) (key / getRowDimension());
            } else {
                j = (int) (key % getColumnDimension());
            }
            nbrElements[j]++;
            if (columnMeans.containsKey(j)) {
                columnMeans.put(j, columnMeans.get(j) + value);
            } else {
                columnMeans.put(j, value);
            }
            entry.put(key, value);
        }
        for (Map.Entry<Integer, Double> e : columnMeans.entrySet()) {
            columnMeans.put(e.getKey(), e.getValue() / nbrElements[e.getKey()]);
        }
        for (Map.Entry<Long, Double> e : entry.entrySet()) {
            final long key = e.getKey();
            final double value = e.getValue();
            final int i, j;
            if (isTransposed()) {
                j = (int) (key / getRowDimension());
                i = (int) (key % getRowDimension());
            } else {
                i = (int) (key / getColumnDimension());
                j = (int) (key % getColumnDimension());
            }
            setEntry(i, j, value - columnMeans.get(j));
        }
    }

    public void cutDataSet(double rateTestSet, double rateValidationSet) {
        if (rateTestSet + rateValidationSet > 100) {
            System.err.println("Invalid rates:" + (rateTestSet + rateValidationSet) + " > 100%");
        } else {
            System.err.println("******************************");
            System.err.println("Initiale dataset size= " + getDataSize());
            System.err.println("******************************");
            int valTestSet = (int) (getDataSize() * rateTestSet / 100);// number of values to remove from the data entry, and put in the test set
            int valValidationSet = (int) (getDataSize() * rateValidationSet / 100);// number of values to remove from the data entry, and put in the validation set
            while (valTestSet > 0) {
                valTestSet--;
                long key = getEntries().getRandomKey();
                double value = getEntries().remove(key);
                testSetEntries.put(key, value);
                final int i, j;
                if (isTransposed()) {
                    j = (int) (key / getRowDimension());
                    i = (int) (key % getRowDimension());
                } else {
                    i = (int) (key / getColumnDimension());
                    j = (int) (key % getColumnDimension());
                }
                double v = rowNonZeroEntries.get(i);
                rowNonZeroEntries.put(i, v - 1);
                v = columnNonZeroEntries.get(j);
                columnNonZeroEntries.put(j, v - 1);
                
            }
            while (valValidationSet > 0) {
                valValidationSet--;
                long key = getEntries().getRandomKey();
                double value = getEntries().remove(key);
                validationSetEntries.put(key, value);
                final int i, j;
                if (isTransposed()) {
                    j = (int) (key / getRowDimension());
                    i = (int) (key % getRowDimension());
                } else {
                    i = (int) (key / getColumnDimension());
                    j = (int) (key % getColumnDimension());
                }
                double v = rowNonZeroEntries.get(i);
                rowNonZeroEntries.put(i, v - 1);
                v = columnNonZeroEntries.get(j);
                columnNonZeroEntries.put(j, v - 1);
            }
            System.err.println("Training dataset size= " + getDataSize());
            System.err.println("Test dataset size= " + getTestSetSize());
            System.err.println("Validation dataset size= " + getValidationSetSize());
            System.err.println("******************************");
        }
    }

    public static DataMatrix createDataMatrix(String file) throws FileNotFoundException, IOException {
        FileInputStream fstream;
        Set<String> listI = new HashSet<>();
        Set<String> listJ = new HashSet();
        fstream = new FileInputStream(file);
        double max = 0;
        // Get the object of DataInputStream
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String str;
        while ((str = br.readLine()) != null) {
            str = str.trim();
            if (str.startsWith("#")) {
                continue;
            }
            if (str.trim().length() == 0) {
                continue;
            }
            StringTokenizer st = new StringTokenizer(str);
            String i = st.nextToken();
            String j = st.nextToken();
            double rating = Double.parseDouble(st.nextToken());
            if (max < rating) {
                max = rating;
            }
            if (!listI.contains(i)) {
                listI.add(i);
            }
            if (!listJ.contains(j)) {
                listJ.add(j);
            }
        }
        return new DataMatrix(file, listI.size(), listJ.size(), max);
    }

    public OpenLongToDoubleHashMap getTestSetEntries() {
        return testSetEntries;
    }

    public int getTestSetSize() {
        return testSetEntries.size();
    }

    public OpenLongToDoubleHashMap getValidationDataEntries() {
        return validationSetEntries;
    }

    public int getValidationSetSize() {
        return validationSetEntries.size();
    }

    public Map<String, Integer> getRowsMapping() {
        return rowsMapping;
    }

    public Map<String, Integer> getColumnsMapping() {
        return columnsMapping;
    }

    public double getEntry(String indexI, String indexJ) {
        int i = rowsMapping.get(indexI);
        int j = columnsMapping.get(indexJ);
        return getEntry(i, j);
    }

    public int getRowNonZeroEntry(int i) {
        if (isTransposed()) {
            return (int) columnNonZeroEntries.get(i);
        } else {
            return (int) rowNonZeroEntries.get(i);
        }
    }

    public int getColumnNonZeroEntry(int j) {
        if (isTransposed()) {
            return (int) rowNonZeroEntries.get(j);
        } else {
            return (int) columnNonZeroEntries.get(j);
        }
    }

    public double getMax() {
        return max;
    }
    
    
}
