package edu.berkeley.nlp.assignments.assign1.student;
import java.util.Random;

public class BitSet {
  int[] bits;
  
  public static void main(String[] args) {
    BitSet b = new BitSet(100000);
    Random r = new Random();
    for (int i = 0; i < 1000000; i++) {
      int j = r.nextInt(1000);
      b.add(j);
      if (!b.contains(j)) {
        System.out.printf("BitSet bug: just inserted element %d but bitset doesn't contain it\n", j);
      }
    }
    
    System.out.println("Done.");
  }
  
  public BitSet(int size) {
    bits = new int[(size + 31) >>> 5];
  }

  public void add(int position) {
    bits[position >>> 5] |= 0x80000000 >>> (position & 31);
  }
  
  public boolean contains(int position) {
    return (bits[position >>> 5] & (0x80000000 >>> (position & 31))) != 0;
  }
  
  public void print() {
    for (int i = 0; i < bits.length; i++) {
      System.out.println(i + ": " + Integer.toBinaryString(bits[i]));
    }
  }
}
