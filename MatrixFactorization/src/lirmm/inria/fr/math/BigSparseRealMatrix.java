/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lirmm.inria.fr.math;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Set;
//import lirmm.inria.fr.peersim.dpmf.Mapping;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.MatrixDimensionMismatchException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SparseRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;

/**
 * Sparse matrix implementation based on an open addressed map.
 *
 * <p>
 * Caveat: This implementation assumes that, for any {@code x}, the equality
 * {@code x * 0d == 0d} holds. But it is is not true for {@code NaN}. Moreover,
 * zero entries will lose their sign. Some operations (that involve {@code NaN}
 * and/or infinities) may thus give incorrect results.
 * </p>
 *
 * @version $Id: OpenMapRealMatrix.java 1569825 2014-02-19 17:19:59Z luc $
 * @since 2.0
 */
public class BigSparseRealMatrix extends AbstractRealMatrix
        implements SparseRealMatrix, Serializable {

    /**
     * Serializable version identifier.
     */
    private static final long serialVersionUID = -5962461716457143437L;
    /**
     * Number of rows of the matrix.
     */
    private final int rows;
    /**
     * Number of columns of the matrix.
     */
    private final int columns;
    /**
     * Storage for (sparse) matrix elements.
     */
    private final OpenLongToDoubleHashMap entries;

    /**
     * Indicate if the matrix is transposed.
     */
    private boolean isTransposed = false;

    /**
     * Build a sparse matrix with the supplied i and column dimensions.
     *
     * @param rowDimension Number of rows of the matrix.
     * @param columnDimension Number of columns of the matrix.
     * @throws NotStrictlyPositiveException if i or column dimension is not
     * positive.
     * @throws NumberIsTooLargeException if the total number of entries of the
     * matrix is larger than {@code Integer.MAX_VALUE}.
     */
    public BigSparseRealMatrix(int rowDimension, int columnDimension)
            throws NotStrictlyPositiveException, NumberIsTooLargeException {
        super(rowDimension, columnDimension);
        long lRow = rowDimension;
        long lCol = columnDimension;
        if (lRow * lCol >= Long.MAX_VALUE) {
            throw new NumberIsTooLargeException(lRow * lCol, Long.MAX_VALUE, false);
        }
        this.rows = rowDimension;
        this.columns = columnDimension;
        this.entries = new OpenLongToDoubleHashMap(0.0);
    }

    /**
     * Build a matrix by copying another one.
     *
     * @param matrix matrix to copy.
     */
    public BigSparseRealMatrix(BigSparseRealMatrix matrix) {
        this.rows = matrix.rows;
        this.columns = matrix.columns;
        this.isTransposed = matrix.isTransposed;
        this.entries = new OpenLongToDoubleHashMap(matrix.entries);
    }

    /**
     * Create a new {@code BigSparseRealMatrix} using the input array as the
     * underlying data array.
     *
     * @param data Data for the new matrix.
     * @throws DimensionMismatchException if {@code d} is not rectangular.
     * @throws NoDataException if {@code d} i or column dimension is zero.
     */
    public BigSparseRealMatrix(final double[][] data)
            throws DimensionMismatchException, NoDataException {
        MathUtils.checkNotNull(data);
        this.entries = new OpenLongToDoubleHashMap(0.0);
        rows = data.length;
        if (rows == 0) {
            throw new NoDataException(LocalizedFormats.AT_LEAST_ONE_ROW);
        }
        columns = data[0].length;
        if (columns == 0) {
            throw new NoDataException(LocalizedFormats.AT_LEAST_ONE_COLUMN);
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                setEntry(i, j, data[i][j]);
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getRowDimension() {
        if (isTransposed) {
            return columns;
        } else {
            return rows;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getColumnDimension() {
        if (isTransposed) {
            return rows;
        } else {
            return columns;
        }
    }

    /**
     * {@inheritDoc}
     *
     * @throws NumberIsTooLargeException if the total number of entries of the
     * matrix is larger than {@code Integer.MAX_VALUE}.
     */
    @Override
    public BigSparseRealMatrix createMatrix(int rowDimension, int columnDimension)
            throws NotStrictlyPositiveException, NumberIsTooLargeException {
        return new BigSparseRealMatrix(rowDimension, columnDimension);
    }

    /**
     * {@inheritDoc}
     *
     * @return
     */
    @Override
    public BigSparseRealMatrix copy() {
        return new BigSparseRealMatrix(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getEntry(int row, int column) throws OutOfRangeException {
        int rowIndex, columnIndex;
        if (isTransposed) {
            rowIndex = column;
            columnIndex = row;
        } else {
            rowIndex = row;
            columnIndex = column;
        }
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        return entries.get(computeKey(rowIndex, columnIndex));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setEntry(int row, int column, double value)
            throws OutOfRangeException {
        int rowIndex, columnIndex;
        if (isTransposed) {
            rowIndex = column;
            columnIndex = row;
        } else {
            rowIndex = row;
            columnIndex = column;
        }
//        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        if (value == 0.0) {
            entries.remove(computeKey(rowIndex, columnIndex));
        } else {
            entries.put(computeKey(rowIndex, columnIndex), value);
        }
    }

    @Override
    public BigSparseRealMatrix transpose() {
        //need to be improved! tool complexe method!
        BigSparseRealMatrix out = this.copy();
        out.isTransposed = !out.isTransposed;
        return out;
    }

    /**
     * Compute the key to access a matrix element
     *
     * @param row i index of the matrix element
     * @param column column index of the matrix element
     * @return key within the map to access the matrix element
     */
    private long computeKey(int row, int column) {
        long lRow = (long) row;
        long lcolumn = (long) column;
        return lRow * columns + lcolumn;
    }

    @Override
    public String toString() {
        DecimalFormat df;
        df = new DecimalFormat();
        df.setMaximumFractionDigits(2); //arrondi à 2 chiffres apres la virgules
        df.setMinimumFractionDigits(2);
        df.setDecimalSeparatorAlwaysShown(true);
        String out = "";
        for (int i = 0; i < getRowDimension(); i++) {
            String o = "";
            for (int j = 0; j < getColumnDimension(); j++) {
                out += df.format(getEntry(i, j)).replaceAll(",", ".") + " ";
                o += df.format(getEntry(i, j)).replaceAll(",", ".") + " ";
            }
            out += "\n";
        }
        return out;
    }

    /**
     * Generate a sparse matrix of random values.
     *
     * @param rowDimension Number of rows of the matrix.
     * @param columnDimension Number of columns of the matrix.
     * @param sparsity Percentage of data in the matrix.
     * @return An instance of BigSparseRealMatrix.
     */
    public static BigSparseRealMatrix randomGenerateMatrix(int rowDimension, int columnDimension, double sparsity) {
        BigSparseRealMatrix m = new BigSparseRealMatrix(rowDimension, columnDimension);
        long lrow = (long) rowDimension;
        long lcolumn = (long) columnDimension;
        long total = (long) (lrow * lcolumn * sparsity / 100);
        Random r = new Random();
        for (long i = 0; i < total; i++) {
            int row = r.nextInt(rowDimension);
            int col = r.nextInt(columnDimension);
            while (m.getEntry(row, col) != 0) {
                row = r.nextInt(rowDimension);
                col = r.nextInt(columnDimension);
            }
            m.setEntry(row, col, r.nextDouble());
        }
        return m;
    }

    /**
     * Generate a matrix of random values.
     *
     * @param rowDimension Number of rows of the matrix.
     * @param columnDimension Number of columns of the matrix.
     * @return An instance of BigSparseRealMatrix.
     */
    public static BigSparseRealMatrix randomGenerateMatrix(int rowDimension, int columnDimension) {
        BigSparseRealMatrix m = new BigSparseRealMatrix(rowDimension, columnDimension);
        Random r = new Random();
        for (int i = 0; i < rowDimension; i++) {
            for (int j = 0; j < columnDimension; j++) {
                m.setEntry(i, j, r.nextDouble());
            }
        }
        return m;
    }

    /**
     * Return the number of non-zero elements in the Matrix
     *
     * @return Number of non-zero elements in the Matrix.
     */
    public int getDataSize() {
        return entries.size();
    }

    /**
     * Return the entries of the Matrix
     *
     * @return Entries of the matrix.
     */
    public OpenLongToDoubleHashMap getEntries() {
        return entries;
    }

    /**
     * Compute the sum of this matrix and {@code m}.
     *
     * @param m Matrix to be added.
     * @return {@code this} + {@code m}.
     * @throws MatrixDimensionMismatchException if {@code m} is not the same
     * size as {@code this}.
     */
    public BigSparseRealMatrix add(BigSparseRealMatrix m)
            throws MatrixDimensionMismatchException {
        MatrixUtils.checkAdditionCompatible(this, m);
        final BigSparseRealMatrix out = new BigSparseRealMatrix(this);
        for (OpenLongToDoubleHashMap.Iterator iterator = m.entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final int row, col;
            if (m.isTransposed) {
                col = (int) (iterator.key() / m.columns);
                row = (int) (iterator.key() - col * m.columns);
            } else {
                row = (int) (iterator.key() / m.columns);
                col = (int) (iterator.key() - row * m.columns);
            }
            out.setEntry(row, col, getEntry(row, col) + iterator.value());
        }
        return out;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void addToEntry(int row, int column, double increment)
            throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        int rowIndex, columnIndex;
        if (isTransposed) {
            rowIndex = column;
            columnIndex = row;
        } else {
            rowIndex = row;
            columnIndex = column;
        }
        final long key = computeKey(rowIndex, columnIndex);
        final double value = entries.get(key) + increment;
        if (value == 0.0) {
            entries.remove(key);
        } else {
            entries.put(key, value);
        }
    }

    @Override
    public double getFrobeniusNorm() {
        double v = 0;
        for (OpenLongToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            v += FastMath.pow(value, 2);
        }
        return FastMath.sqrt(v);
    }

    @Override
    public BigSparseRealMatrix scalarMultiply(double d) {
        BigSparseRealMatrix out = new BigSparseRealMatrix(rows, columns);
        out.isTransposed = isTransposed;
        for (OpenLongToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            final long key = iterator.key();
            final int i, j;
            if (isTransposed) {
                j = (int) (key / columns);
                i = (int) (key % columns);
            } else {
                i = (int) (key / columns);
                j = (int) (key % columns);
            }
            out.setEntry(i, j, value * d);
        }
        return out;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public BigSparseRealMatrix subtract(final RealMatrix m)
            throws MatrixDimensionMismatchException {
        try {
            return subtract((BigSparseRealMatrix) m);
        } catch (ClassCastException cce) {
            return (BigSparseRealMatrix) super.subtract(m);
        }
    }

    /**
     * Subtract {@code m} from this matrix.
     *
     * @param m Matrix to be subtracted.
     * @return {@code this} - {@code m}.
     * @throws MatrixDimensionMismatchException if {@code m} is not the same
     * size as {@code this}.
     */
    public BigSparseRealMatrix subtract(BigSparseRealMatrix m)
            throws MatrixDimensionMismatchException {
        MatrixUtils.checkAdditionCompatible(this, m);
        final BigSparseRealMatrix out = new BigSparseRealMatrix(this);
        for (OpenLongToDoubleHashMap.Iterator iterator = m.entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final int row, col;
            if (m.isTransposed) {
                col = (int) (iterator.key() / m.columns);
                row = (int) (iterator.key() % m.columns);
            } else {
                row = (int) (iterator.key() / m.columns);
                col = (int) (iterator.key() % m.columns);
            }
            out.setEntry(row, col, getEntry(row, col) - iterator.value());
        }
        return out;
    }

    /**
     * Postmultiply this matrix by {@code m}.
     *
     * @param m Matrix to postmultiply by.
     * @return {@code this} * {@code m}.
     * @throws DimensionMismatchException if the number of rows of {@code m}
     * differ from the number of columns of {@code this} matrix.
     * @throws NumberIsTooLargeException if the total number of entries of the
     * product is larger than {@code Integer.MAX_VALUE}.
     */
    public BigSparseRealMatrix multiply(BigSparseRealMatrix m)
            throws DimensionMismatchException, NumberIsTooLargeException {
        // Safety check.
        MatrixUtils.checkMultiplicationCompatible(this, m);
        final int outCols = m.getColumnDimension();
        BigSparseRealMatrix out = new BigSparseRealMatrix(getRowDimension(), outCols);
        for (OpenLongToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            final long key = iterator.key();
            final int i, k;
            if (isTransposed) {
                k = (int) (key / columns);
                i = (int) (key % columns);
            } else {
                i = (int) (key / columns);
                k = (int) (key % columns);
            }
            for (int j = 0; j < outCols; ++j) {
                final long rightKey;
                if (m.isTransposed) {
                    rightKey = m.computeKey(j, k);
                } else {
                    rightKey = m.computeKey(k, j);
                }
                if (m.entries.containsKey(rightKey)) {
                    final long outKey = out.computeKey(i, j);
                    final double outValue
                            = out.entries.get(outKey) + value * m.entries.get(rightKey);
                    if (outValue == 0.0) {
                        out.entries.remove(outKey);
                    } else {
                        out.entries.put(outKey, outValue);
                    }
                }
            }
        }
        return out;
    }

    /**
     * {@inheritDoc}
     *
     * @throws NumberIsTooLargeException if {@code m} is an
     * {@code OpenMapRealMatrix}, and the total number of entries of the product
     * is larger than {@code Integer.MAX_VALUE}.
     */
    @Override
    public RealMatrix multiply(final RealMatrix m)
            throws DimensionMismatchException, NumberIsTooLargeException {
        try {
            return multiply((BigSparseRealMatrix) m);
        } catch (ClassCastException cce) {
            MatrixUtils.checkMultiplicationCompatible(this, m);
            final int outCols = m.getColumnDimension();
            final BigSparseRealMatrix out = new BigSparseRealMatrix(rows, outCols);
            for (OpenLongToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
                iterator.advance();
                final double value = iterator.value();
                final long key = iterator.key();
                final int i, k;
                if (isTransposed) {
                    k = (int) (key / columns);
                    i = (int) (key % columns);
                } else {
                    i = (int) (key / columns);
                    k = (int) (key % columns);
                }
                for (int j = 0; j < outCols; ++j) {
                    out.addToEntry(i, j, value * m.getEntry(k, j));
                }
            }
            return out;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void multiplyEntry(int row, int column, double factor)
            throws OutOfRangeException {
        MatrixUtils.checkRowIndex(this, row);
        MatrixUtils.checkColumnIndex(this, column);
        int rowIndex, columnIndex;
        if (isTransposed) {
            rowIndex = column;
            columnIndex = row;
        } else {
            rowIndex = row;
            columnIndex = column;
        }
        final long key = computeKey(rowIndex, columnIndex);
        final double value = entries.get(key) * factor;
        if (value == 0.0) {
            entries.remove(key);
        } else {
            entries.put(key, value);
        }
    }

    /**
     * Subtract {@code A*B} from this matrix.
     *
     * @param A Latent feature matrix 1
     * @param B Latent feature matrix 2
     * @return {@code this} - {@code A} * {@code B}.
     */
    public BigSparseRealMatrix specialOperation(BigSparseRealMatrix A, BigSparseRealMatrix B) {
        MatrixUtils.checkMultiplicationCompatible(A, B);
        MatrixUtils.checkAdditionCompatible(this, new BigSparseRealMatrix(A.getRowDimension(), B.getColumnDimension()));
        BigSparseRealMatrix out = new BigSparseRealMatrix(getRowDimension(), getColumnDimension());
        for (OpenLongToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            final long key = iterator.key();
            final int i, j;
            if (isTransposed) {
                j = (int) (key / columns);
                i = (int) (key % columns);
            } else {
                i = (int) (key / columns);
                j = (int) (key % columns);
            }
            double v = value - A.getRowVector(i).dotProduct(B.getColumnVector(j));
            out.setEntry(i, j, v);
        }
        return out;
    }

    public boolean isTransposed() {
        return isTransposed;
    }

    public void print() {
        for (OpenLongToDoubleHashMap.Iterator iterator = entries.iterator(); iterator.hasNext();) {
            iterator.advance();
            final double value = iterator.value();
            final long key = iterator.key();
            final int i, j;
            i = (int) (key / columns);
            j = (int) (key % columns);
            System.out.println(i + "\t" + j + "\t" + value);
        }
    }

    /**
     * Add the vector vec to the collumn c of the current matrix
     *
     *
     * @param m
     * @param mappings
     * @return {@code this} + {@code m}.
     * @throws MatrixDimensionMismatchException if {@code m} is not the same
     * size as {@code this}.
     */
//    public BigSparseRealMatrix addToColumn(BigSparseRealMatrix m, Set< Mapping> mappings)
//            throws MatrixDimensionMismatchException {
//        final BigSparseRealMatrix out = new BigSparseRealMatrix(this);
//        for (Mapping map : mappings) {
//            for (int row = 0; row < getRowDimension(); row++) {
//                out.setEntry(row, map.getMainNodeIndice(), getEntry(row, map.getMainNodeIndice()) + m.getEntry(row, map.getCurrentNodeIndice()));
//            }
//        }
//        return out;
//    }

}
