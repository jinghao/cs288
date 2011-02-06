package edu.berkeley.nlp.assignments.assign1.student;

import java.util.Arrays;
import java.util.List;

public class ExactLm extends MultilevelLm {	
	public ExactLm(Iterable<List<String>> sentenceCollection) {
		System.out.println("Building ExactLm... ");
		processSentences(sentenceCollection);
		compact();
		
		System.out.println("Number of word types (unigrams): " + bigrams.length);
		System.out.println("Number of bigrams/trigrams: " + num_bigrams + " / " + num_trigrams);
	}
	
	private void compact() {
		System.out.println("Compacting data structure.");
		long start = System.nanoTime();
		
		long[][] new_bigrams = new long[indexer.size()][];
		
		for (int word1 = 0; word1 < new_bigrams.length; word1++) {
			if (bigrams_count[word1] > 0) {
				new_bigrams[word1] = new long[bigrams_count[word1]];
				int j = 0;
				for (long entry : bigrams[word1]) {
					if (entry != 0) {
						new_bigrams[word1][j++] = entry;
					}
				}
				Arrays.sort(new_bigrams[word1]);
			}
			bigrams[word1] = null; // make gc happy
		}
		
		bigrams = new_bigrams;
		
		// make gc happy
		bigrams_count = null;
		
		System.out.println("Done compacting. Took " + (System.nanoTime() - start) / 1000000000. + " seconds.");
	}
}