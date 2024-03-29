import torch
import numpy as np
import pickle, json


def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)


@torch.no_grad()
def knn_classifier(
        train_features, 
        train_labels, 
        test_features, 
        test_labels, 
        k, T, 
        offset=0, # if the train_features are the same as the test features, offset should be set to 1
        num_classes=1000,
        use_cuda=True):

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes)
    if use_cuda:
        retrieval_one_hot = retrieval_one_hot.cuda()

    ###########################################################
    # The indices of the topk neighbors for each image
    nn_inds = []
    ###########################################################
    
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)

        ###########################################################
        # collect the indices of the topk neighbors
        distances, indices = similarity.topk(k+offset, largest=True, sorted=True)
        distances = distances[:,offset:]
        indices = indices[:,offset:]
        nn_inds.append(indices.cpu().data.numpy())
        ###########################################################

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    
    nn_inds = np.concatenate(nn_inds, 0)
    # print(nn_inds.shape)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5, nn_inds