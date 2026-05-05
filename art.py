import numpy as np

class ART1:
    def __init__(self, num_features, num_clusters, vigilance=0.5):
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.vigilance = vigilance
        
        # Initialize weights
        self.bottom_up = np.ones((num_clusters, num_features)) / (1 + num_features)
        self.top_down = np.ones((num_clusters, num_features))

    def train(self, data):
        for sample in data:
            self._process_sample(sample)

    def _process_sample(self, sample):
        # Step 1: Compute matching scores
        scores = np.dot(self.bottom_up, sample)

        # Step 2: Find best matching cluster
        while True:
            winner = np.argmax(scores)

            # Step 3: Compute match (vigilance test)
            top_down_weight = self.top_down[winner]
            intersection = np.minimum(sample, top_down_weight)
            match = np.sum(intersection) / np.sum(sample)

            if match >= self.vigilance:
                # Resonance occurs → update weights
                self._update_weights(winner, sample)
                print(f"Pattern {sample} assigned to cluster {winner}")
                break
            else:
                # Reset this cluster and try next
                scores[winner] = -1

                if np.all(scores == -1):
                    print("No suitable cluster found.")
                    break

    def _update_weights(self, winner, sample):
        # Update top-down weights
        self.top_down[winner] = np.minimum(self.top_down[winner], sample)

        # Update bottom-up weights
        self.bottom_up[winner] = self.top_down[winner] / (0.5 + np.sum(self.top_down[winner]))


# 🔹 Example Usage
if __name__ == "__main__":
    # Binary input patterns
    data = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])

    art = ART1(num_features=4, num_clusters=3, vigilance=0.6)
    art.train(data)
