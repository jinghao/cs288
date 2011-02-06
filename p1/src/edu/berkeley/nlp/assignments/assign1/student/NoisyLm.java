package edu.berkeley.nlp.assignments.assign1.student;

import java.util.Arrays;
import java.util.List;
import edu.berkeley.nlp.util.CollectionUtils;

public class NoisyLm extends MultilevelLm {
  private int BITSET_SIZE;
  private int NUM_HASH = 3;
  private int[] bitset_hashcodes = new int[NUM_HASH];
  private BitSet singletons;
  
  public NoisyLm(Iterable<List<String>> sentenceCollection) {
    System.out.println("Building NoisyLm... ");
    
    for (int i = 0; i < NUM_HASH; i++) {
      bitset_hashcodes[i] = getHashCode();
    }
    
    processSentences(sentenceCollection);
    compact();
    
    System.out.println("Number of word types (unigrams): " + bigrams.length);
    System.out.println("Number of bigrams/trigrams: " + num_bigrams + " / " + num_trigrams);
  }

  private void compact() {
    System.out.println("Compacting data structure.");
    long start = System.nanoTime();
    
    long[][] new_bigrams = new long[indexer.size()][];
    
    // find out number of singletons
    int num_singletons = 0;
    for (int word1 = 0; word1 < new_bigrams.length; word1++) {
      if (bigrams_count[word1] > 0) {
        for (long entry : bigrams[word1]) {
          int count = (int)(entry & mask_25);
          if (count == 1) {
            ++num_singletons;
          }
        }
      }
    }

    // pick the optimal size given the number of hash functions and elements, and make it multiple of 32
    BITSET_SIZE = ((int)((NUM_HASH * num_singletons) / Math.log(2))) & ~0x1F;
    System.out.printf("Found %d singletons... Making bitset of size %d bits\n", num_singletons, BITSET_SIZE);
    
    // initializing singletons bitset
    singletons = new BitSet(BITSET_SIZE);
    
    for (int word1 = 0; word1 < new_bigrams.length; word1++) {
      if (bigrams_count[word1] > 0) {
        int next_entry = 0;
        for (long entry : bigrams[word1]) {
          if (entry != 0) {
            int count = (int)(entry & mask_25);
            if (count == 1) {
              int word2 = (int)(entry >>> 44) & mask_19;
              int word3 = (int)(entry >>> 25) & mask_19;
              bitsetInsert(word1, word2, word3);
            } else {
              bigrams[word1][next_entry++] = entry;
            }
          }
        }

        new_bigrams[word1] = CollectionUtils.copyOf(bigrams[word1], next_entry);
        Arrays.sort(new_bigrams[word1]);
      }
      
      bigrams[word1] = null; // make gc happy
    }
    
    bigrams = new_bigrams;
    bigrams_count = null;

    System.out.println("Done compacting. Took " + (System.nanoTime() - start) / 1000000000. + " seconds.");
  }

  private int bitset_hash(int word1, int word2, int word3, int i) {
    return (((hashcodes[word3] >>> 2) ^ (hashcodes[word2] >>> 1) ^ hashcodes[word1]) ^ bitset_hashcodes[i]) % BITSET_SIZE;
  }
  
  long defaultCount(int word1, int word2, int word3) {
    return bitsetContains(word1, word2, word3) ? 1 : 0;
  }
  
  private void bitsetInsert(int word1, int word2, int word3) {
    for (int i = 0; i < NUM_HASH; i++) {
      singletons.add(bitset_hash(word1, word2, word3, i));
    }
  }
  
  private boolean bitsetContains(int word1, int word2, int word3) {
    for (int i = 0; i < NUM_HASH; i++) {
      if (!singletons.contains(bitset_hash(word1, word2, word3, i))) {
        return false;
      }
    }
    
    return true;
  }
}