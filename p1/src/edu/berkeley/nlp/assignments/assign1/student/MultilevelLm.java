package edu.berkeley.nlp.assignments.assign1.student;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.StringIndexer;

abstract class MultilevelLm implements NgramLanguageModel {
  final int INITIAL_WORDS = 500000;
  final double d = 0.75;

  static final int mask_19 = ~0 >>> (32 - 19);
  static final int mask_25 = ~0 >>> (32 - 25);
  static final long mask_38 = (~(long)0) >>> (64 - 38);
  
  static Random random = new Random(0xDEADD0D0);

  StringIndexer indexer = EnglishWordIndexer.getIndexer();
  int START = indexer.addAndGetIndex(NgramLanguageModel.START);
  int STOP = indexer.addAndGetIndex(NgramLanguageModel.STOP);
  int EMPTY = indexer.addAndGetIndex("<w>");
  int SUFFIX = indexer.addAndGetIndex("<s>");

  int[] bigrams_count = new int[INITIAL_WORDS]; // cleared
  
  long[][] bigrams = new long[INITIAL_WORDS][];
  int[] unigram_counts = new int[INITIAL_WORDS]; // c(w)
  int[] prefix_count = new int[INITIAL_WORDS]; // t(*,w)
  int[] suffix_count = new int[INITIAL_WORDS]; // t(w,*)
  int[] hashcodes = new int[INITIAL_WORDS];
  
  int num_bigrams = 0;
  int num_trigrams = 0;
  
  /**
   * Cache parameters and variables
   */
  int CACHE_SIZE = 100000;
  long[] cache_keys = new long[CACHE_SIZE];
  long[] cache_values = new long[CACHE_SIZE];
  
  void processSentences(Iterable<List<String>> sentenceCollection) {
    int sent = 0;
    
    long start = System.nanoTime();
    
    for (int i = 0; i < bigrams.length; i++) {
      bigrams[i] = new long[5];
      hashcodes[i] = getHashCode();
    }

    for (List<String> sentence : sentenceCollection) {
      sent++;
      if (sent % 100000 == 0) {
        System.out.printf("%d,%f,%d,%d,%d\n", 
          sent, 
          (System.nanoTime() - start) / 1000000000., 
          indexer.size(),
          num_bigrams,
          num_trigrams
          );
      }
      
      int[] stoppedSentence = new int[sentence.size() + 2];
      stoppedSentence[0] = START;
      stoppedSentence[stoppedSentence.length - 1] = STOP;
      
      int i = 0;
      
      for (String word : sentence) {
        ++unigram_counts[stoppedSentence[++i] = indexer.addAndGetIndex(word)];
      }
      
      for (i = 0; i < stoppedSentence.length - 1; ++i) {
        // first time we've seen word1 => new "* word2"
        if (bigrams_count[stoppedSentence[i]] == 0) {
          ++prefix_count[stoppedSentence[i + 1]];
        }

        if (addWord(stoppedSentence[i], stoppedSentence[i + 1], EMPTY)) {
          // didn't find "word1 word2" => new "word1 *"
          ++suffix_count[stoppedSentence[i]];
          ++num_bigrams;
        }
        
        if (i < stoppedSentence.length - 2) {
          // add stoppedSentence[i ... i+2]
          if (addWord(stoppedSentence[i], stoppedSentence[i + 1], stoppedSentence[i + 2])) {
            // didn't find "word1 word2 word3" => new "word1 word2 *"
            addWord(stoppedSentence[i], stoppedSentence[i + 1], SUFFIX);
            
            ++num_trigrams;
          }
        }
      }
    }
    
    System.out.println("Done building language model. Took " + (System.nanoTime() - start) / 1000000000. + " seconds.");
  }
  
  public int getOrder() {
    return 3;
  }
  
  public double getNgramLogProbability(int[] ngram, int from, int to) {
    return Math.log(getNgramProbability(ngram, from, to));
  }

  // P(ngram[to-1] | ngram[from..to-2])
  private double getNgramProbability(int[] ngram, int from, int to) {
    long c_uw, c_u, t_u;
    
    double prob;
    
    switch (to - from) {
    case 3: // Pr(ngram[from + 2] | ngram[from..from + 1]
      int word1 = ngram[from];
      int word2 = ngram[from + 1];
      
      prob = getNgramProbability(ngram, from + 1, to);

      if (word1 >= bigrams.length || word2 >= bigrams.length) {
        return prob;
      }

      int hashcode = hash(word1, word2);
      long bigram_key = bigram_key(word1, word2);
      
      c_u = getCountInternal(bigram_key, hashcode, word1, word2, EMPTY); // c(w1,w2,*)
      
      if (c_u == 0) {
        return prob;
      }
      
      c_uw = ngram[from + 2] >= bigrams.length ? 0 : getCountInternal(bigram_key, hashcode, word1, word2, ngram[from + 2]); // c(w1,w2,w3)
      t_u = getCountInternal(bigram_key, hashcode, word1, word2, SUFFIX); // t(w1,w2,*)

      break;
    case 2: // Pr(ngram[from + 1] | ngram[from]
      prob = getNgramProbability(ngram, from + 1, to);
      
      word1 = ngram[from];
      word2 = ngram[from + 1];
      
      if (word1 >= bigrams.length || (c_u = unigram_counts[word1]) == 0) { // c(w1)
        return prob;
      }
      
      c_uw = getCountInternal(bigram_key(word1, word2), hash(word1, word2), word1, word2, EMPTY); // c(w1,w2)
      
      t_u = suffix_count[word1]; // t(w1, *)
      break;
    case 1: // Pr(ngram[from])
      c_uw = prefix_count[ngram[from]]; // t(*, w1)
      c_u = num_bigrams; // num bigrams t(*, *)
      
      return Math.max(d, c_uw) / c_u;
    default:
      System.err.printf("Asked for Ngram probability with from..to = %d..%d\n", from, to);
      throw new AssertionError();
    }

    // c_u is actually 0. False positive for c_u
    if (t_u == 0 && c_u == 1)
      return prob;
    
    prob = (Math.max(0, c_uw - d) + d * t_u * prob) / c_u;
    
    assert (prob > 0 && prob < 1) : "Probability should be in (0,1)";
    
    return prob;
  }

  public long getCount(int[] ngram) {
    switch (ngram.length) {
    case 3:
      return getCount(ngram[0], ngram[1], ngram[2]); // c(w1,w2,w3)
    case 2:
      return getCount(ngram[0], ngram[1], EMPTY); // c(w1,w2,*)
    case 1:
      return unigram_counts[ngram[0]]; // c(w1)
    default:
      return 0;
    }
  }
  
  long getCount(int word1, int word2, int word3) {
    // one of the word we've never seen before
    if (word1 >= bigrams.length || word2 >= bigrams.length || word3 >= bigrams.length) {
      return 0;
    }
    
    return getCountInternal(bigram_key(word1, word2), hash(word1, word2), word1, word2, word3);
  }
  
  long cache_accesses = 0, cache_hits = 0;
  
  long getCountInternal(long key, int hash, int word1, int word2, int word3) {
    hash = ((hash >>> 1) ^ hashcodes[word3]) % CACHE_SIZE;
    key = (key << 19) | word3;
    
    // ++cache_accesses;
    if (cache_keys[hash] == key) {
      // ++cache_hits;
      return cache_values[hash];
    }
      
    int index = indexOf(bigrams[word1], bigram_key(word2, word3));

    cache_keys[hash] = key;
    cache_values[hash] = index >= 0 ? (bigrams[word1][index] & mask_25) : defaultCount(word1, word2, word3);
    
    return cache_values[hash];
  }
  
  long defaultCount(int word1, int word2, int word3) {
    return 0;
  }
  
  int getHashCode() {
    return 1 + random.nextInt(Integer.MAX_VALUE);
  }
  
  // binary search bigram_table for the element whose upper 39 bits match key
  int indexOf(long[] bigram_table, long key) {
    int min = 0;
    int max = bigram_table.length - 1;
    
    long upper = (key + 1) << 25;
    key <<= 25; // lower boundary
    
    // find the element in between 'key' and 'upper' (exclusive or inclusive doesn't matter)
    while (min <= max) {
      int mid = min + (max - min) / 2;
      
      if (bigram_table[mid] > upper) {
        // min < key < upper < bigram_table[mid]
        max = mid - 1;
      } else if (bigram_table[mid] > key) {
        // min < key < bigram_table[mid] < upper
        return mid;
      } else {
        // min < bigram_table[mid] < key < upper
        min = mid + 1;
      }
    }
    
    return -1; // didn't find it
  }
  
  int findIndex(long[] table, int index, long key) {
    index = index % table.length;
    
    while (table[index] != 0 && (table[index] >>> 25) != key) {
      index = (index + 1) % table.length;
    }
    
    return index;
  }
  
  // equivalent to table[findIndex(table, index, UNSEEN_KEY)] = value but without the key shift/check
  void setFirstEmptySlot(long[] table, int index, long value) {
    index = index % table.length;
    
    while (table[index] != 0) {
      index = (index + 1) % table.length;
    }
    
    table[index] = value;
  }

  boolean addWord(int word1, int word2, int word3) {
    long[] bigram_table = bigrams[word1];
    long key = bigram_key(word2, word3);
    
    int index = hash(word2, word3);
    int i = findIndex(bigram_table, index, key);

    // new entry
    if (bigram_table[i] == 0) {
      bigram_table[i] = (key << 25) | 1;
      ++bigrams_count[word1];

      // resize?
      if (bigrams_count[word1] >= bigram_table.length / 2) {
        int last_index = 0;
        for (int j = 0; last_index < bigrams_count[word1]; ++j) {
          long entry = bigram_table[j];
          if (entry != 0) {
            bigram_table[last_index++] = ((entry & mask_25) << 38) | (entry >>> 25);
          }
        }
        
        Arrays.sort(bigram_table, 0, last_index);
        // assert bigrams_count[word1] == last_index;
        
        long[] new_table = new long[5 * bigram_table.length / 3];
        
        while (last_index > 0) {
          --last_index;
          
          long entry_key = bigram_table[last_index] & mask_38;
          long entry = (bigram_table[last_index] >>> 38) | (entry_key << 25);
          
          int entry_word2 = (int) (entry_key >>> 19);
          int entry_word3 = (int) (entry_key & mask_19);
          int entry_index = hash(entry_word2, entry_word3);
          
          setFirstEmptySlot(new_table, entry_index, entry);
        }

        bigrams[word1] = new_table;
      }
      
      return true;
    } else {
      // Found it: (bigram_table[i] >>> 25) == key
      ++bigram_table[i];
      
      return false;
    }
  }
  
  int hash(int word2, int word3) {
    return (hashcodes[word2] >>> 1) ^ (hashcodes[word3]);
  }
  
  long bigram_key(long word2, long word3) {
    return (word2 << 19) | word3;
  }
}