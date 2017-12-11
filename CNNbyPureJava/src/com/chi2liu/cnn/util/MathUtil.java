/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.util
 * @fileName: MathUtil.java
 */
package com.chi2liu.cnn.util;

import java.util.function.Function;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date��2017��12��10�� ����2:12:41
 * @className: MathUtil.java
 * @version: 1.0
 */
public class MathUtil {

	public static double[][] trans(final double[][] doubles, Function<Double, Double> function) {
        double[][] trans = new double[doubles.length][doubles[0].length];
        for (int i = 0; i < doubles.length; i++) {
            for (int j = 0; j < doubles[0].length; j++) {
                trans[i][j] = function.apply(doubles[i][j]);
            }
        }
        return trans;
    }

    public static double[][] trans(final double[][] doubles1, final double[][] doubles2, Function<Double, Function<Double, Double>> function) {
        if (doubles1.length != doubles2.length) {
            throw new RuntimeException("two matrix length not equal");
        }
        if (doubles1[0].length != doubles2[0].length) {
            throw new RuntimeException("two matrix length not equal");
        }
        double[][] trans = new double[doubles1.length][doubles1[0].length];
        for (int i = 0; i < doubles1.length; i++) {
            for (int j = 0; j < doubles1[0].length; j++) {
                trans[i][j] = function.apply(doubles1[i][j]).apply(doubles2[i][j]);
            }
        }
        return trans;
    }

    /**
     * ����Ŵ�
     *
     * @param matrix
     * @param scale
     * @return
     */
    public static double[][] scale(final double[][] matrix, final Size scale) {
        final int m = matrix.length;
        int n = matrix[0].length;
        final double[][] outMatrix = new double[m * scale.getWidth()][n * scale.getHeight()];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int ki = i * scale.getWidth(); ki < (i + 1) * scale.getWidth(); ki++) {
                    for (int kj = j * scale.getHeight(); kj < (j + 1) * scale.getHeight(); kj++) {
                        outMatrix[ki][kj] = matrix[i][j];
                    }
                }
            }
        }
        return outMatrix;
    }


    public static double[][] transRotate180(double[][] doubles) {
        doubles = trans(doubles, v -> v);
        int m = doubles.length;
        int n = doubles[0].length;
        // ���жԳƽ��н���
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n / 2; j++) {
                double tmp = doubles[i][j];
                doubles[i][j] = doubles[i][n - 1 - j];
                doubles[i][n - 1 - j] = tmp;
            }
        }
        // ���жԳƽ��н���
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m / 2; i++) {
                double tmp = doubles[i][j];
                doubles[i][j] = doubles[m - 1 - i][j];
                doubles[m - 1 - i][j] = tmp;
            }
        }
        return doubles;
    }


    public static Double sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    private static void p(double[][] matrix) {
        for (int i = 0; i < matrix[0].length; i++) {
            System.out.print("[");
            for (int j = 0; j < matrix.length; j++) {
                System.out.print((j > 0 ? " " : "") + matrix[j][i]);
            }
            System.out.println("]");
        }
    }

    /**
     * ����fullģʽ�ľ��
     * ��νfullģʽ�������ԭ��������ܲ���0ֵ(ͼƬ���ױ�)��ʹ������ƶ����ƶ���ԭ����ı�Ե��Ȼ��Ը���չ������о��
     * ���磺
     * <pre>
     *         ԭ����                 kernel                             ��չ����
     *                                                   [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     * [1.1 1.2 1.1 1.2 1.1]                             [0.0 0.0 1.1 1.2 1.1 1.2 1.1 0.0 0.0]
     * [1.2 0.0 1.0 0.0 1.2]       [1.0 1.0 2.0]         [0.0 0.0 1.2 0.0 1.0 0.0 1.2 0.0 0.0]
     * [1.1 0.0 1.0 0.0 1.1]       [0.0 1.0 1.0]         [0.0 0.0 1.1 0.0 1.0 0.0 1.1 0.0 0.0]
     * [1.2 1.1 1.2 1.1 1.2]                             [0.0 0.0 1.2 1.1 1.2 1.1 1.2 0.0 0.0]
     *                                                   [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     * </pre>
     *
     * @param matrix
     * @param kernel
     * @return
     */
    public static double[][] fullConvolutional(double[][] matrix,
                                               double[][] kernel) {
        // ��չ����
    	//kernel = MathUtil.transRotate180(kernel);
        final double[][] extend = new double[matrix.length + 2 * (kernel.length - 1)][matrix[0].length + 2 * (kernel[0].length - 1)];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++)
                extend[i + kernel.length - 1][j + kernel[0].length - 1] = matrix[i][j];
        }
        return validConvolutional(extend, kernel);
    }

    /**
     * validate ���
     * ���㷽����
     * * ������˾���ŵ�Դ�������Ͻ��ϣ�Ȼ��һ��һ�е������ƶ���һ��һ�������ƶ�ֱ����Ե��
     * * ÿ�ƶ�һ����λ�ã���������˵�ÿ��Ԫ���븲�Ǿ�����Ӧλ���ϵ�ÿ��Ԫ�س˻�����ͺ���Ϊ��������
     * <pre>
     *          Դ���󡡡���������������������ˡ���������������������������
     *     [1.0 1.0 1.0 1.0]                             [2.0 3.0 3.0]
     *     [0.0 0.0 1.0 1.0]         [1.0 1.0]           [1.0 2.0 2.0]
     *     [0.0 1.0 1.0 0.0]         [0.0 1.0]           [2.0 3.0 1.0]
     *     [0.0 1.0 1.0 0.0]
     *
     * </pre>
     *
     * @param doubles
     * @param kernel
     * @return
     */
    public static double[][] validConvolutional(double[][] doubles, double[][] kernel) {
        //kernel = MathUtil.transRotate180(kernel);
    	double[][] convolution = new double[doubles.length - kernel.length + 1][doubles[0].length - kernel[0].length + 1];
        for (int i = 0; i < convolution.length; i++) {
            for (int j = 0; j < convolution[0].length; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < kernel.length; ki++) {
                    for (int kj = 0; kj < kernel[0].length; kj++) {
                        sum += doubles[i + ki][j + kj] * kernel[ki][kj];
                    }
                }
                convolution[i][j] = sum;
            }
        }
        return convolution;
    }
}
