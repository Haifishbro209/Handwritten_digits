import random

import backpropagation as bp


def run_random_predictions(n: int = 100, use_test_data: bool = False, seed: int | None = None) -> None:

	ds = bp.test_data if use_test_data else bp.dataset
	print(len(ds))
	random.seed(seed)
	labelarr = []
	predarr = []
	indices = random.sample(range(len(ds)), n)
    
	correct = 0
	for idx in indices:
		img, label = ds[idx]
		img_flat = img.flatten().tolist()
		output, *_ = bp.forwardpass(img_flat, label)
		pred = max(range(len(output)), key=lambda i: output[i])
		#print(f"idx={idx:5d}  pred={pred}  actual={label}")
		labelarr.append(label)
		predarr.append(pred)
		
		if pred == label:
			correct += 1

	acc = correct / n
	for i in range(10):
		print(f"{i} pred : {predarr.count(i)}")
		print(f"{i} label: {labelarr.count(i)}")
        
	print(f"\nSample accuracy: {correct}/{n} = {acc:.2%}")
    

if __name__ == "__main__":
	run_random_predictions(n=10000, use_test_data=True)
