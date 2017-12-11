/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.util
 * @fileName: Size.java
 */
package com.chi2liu.cnn.util;

import java.io.Serializable;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午2:41:14
 * @className: Size.java
 * @version: 1.0
 */
public class Size implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private final int width;
	private final int height;
	
	public Size(final int width, final int height) {
		this.width = width;
		this.height = height;
	}
	
	public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
    
    public Size subtract(Size size, int append) {
    	int newWidth = this.width - size.width + append;
    	int newHeight = this.height - size.width + append;
    	return new Size(newWidth, newHeight);
    }
    
    public Size divide(Size scaleSize) {
    	int newWidth = this.width / scaleSize.width;
    	int newHeight = this.height / scaleSize.height;
    	if(newWidth * scaleSize.width != this.width || newHeight * scaleSize.height != this.height)
    		throw new RuntimeException("invalidate scale size");
    	return new Size(newWidth, newHeight);
    }
	

}
