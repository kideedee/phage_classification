import pandas as pd
from Bio import SeqIO
from torch_geometric.loader import DataLoader

from experiment.graph import gcn_model
from experiment.graph.Biodata import GraphDatasetInMem, create_dataset
from experiment.graph.gcn_model import GCNModel


def load_data(fasta_file) -> pd.DataFrame:
    sequences = []
    labels = []

    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(seq_record.seq))
        if seq_record.id.split("_")[2] == 'temperate':
            labels.append(0)
        else:
            labels.append(1)
    return pd.DataFrame({"sequence": sequences, "target": labels})


if __name__ == '__main__':

    for fold in range(1, 6):
        for i in range(4):
            if i == 0:
                group = "100_400"
            elif i == 1:
                group = "400_800"
            elif i == 2:
                group = "800_1200"
            else:
                group = "1200_1800"

            print("=" * 100)
            print(f"Start group: {group}, fold: {fold}")

            train_file = f"E:\master\\final_project\data\my_data\\fasta\\{group}\\{fold}\\train\data.fa"
            test_file = f"E:\master\\final_project\data\my_data\\fasta\\{group}\\{fold}\\test\data.fa"

            create_dataset(train_file, f"train")
            create_dataset(test_file, f"test")

            train_ds = GraphDatasetInMem(
                pnode_feature=None,  # These can be None since we're loading
                fnode_feature=None,
                other_feature=None,
                edge=None,
                graph_label=None,
                root="./train"  # Path where your processed data is stored
            )

            test_ds = GraphDatasetInMem(
                pnode_feature=None,  # These can be None since we're loading
                fnode_feature=None,
                other_feature=None,
                edge=None,
                graph_label=None,
                root="./test"  # Path where your processed data is stored
            )

            k = 3
            d = k - 1
            model = GCNModel(k=k, d=d)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])
            # valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, follow_batch=['x_src', 'x_dst'])
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])

            gcn_model.train(train_loader, test_loader, model)

    # train_ds = Biodata(df=train_df, output_name="train.dataset").encode(save_dataset=False)
    # valid_ds = Biodata(df=valid_df, output_name="valid.dataset").encode(save_dataset=False)
    # test_ds = Biodata(df=test_df, output_name="test.dataset").encode(save_dataset=False)

    # train_file = "E:\master\\final_project\data\my_data\start_here\\fold_1\\train\data.csv"
    # test_file = "E:\master\\final_project\data\my_data\start_here\\fold_1\\test\data.csv"
    #
    # train_df = pd.read_csv(train_file)
    # train_df.columns.values[1] = 'label'
    # test_df = pd.read_csv(test_file)
    # test_df.columns.values[1] = 'label'

    # train_ds = Biodata(df=train_df, output_name="train.dataset", k=k, d=d).encode(save_dataset=False)
    # test_ds = Biodata(df=test_df, output_name="test.dataset", k=k, d=d).encode(save_dataset=False)

    # train_loader = DataLoader(train_df, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_df, batch_size=64, shuffle=True)

    # train_ds = GraphDatasetInMem(
    #     pnode_feature=None,  # These can be None since we're loading
    #     fnode_feature=None,
    #     other_feature=None,
    #     edge=None,
    #     graph_label=None,
    #     root="./train"  # Path where your processed data is stored
    # )
    #
    # test_ds = GraphDatasetInMem(
    #     pnode_feature=None,  # These can be None since we're loading
    #     fnode_feature=None,
    #     other_feature=None,
    #     edge=None,
    #     graph_label=None,
    #     root="./test"  # Path where your processed data is stored
    # )
    #
    # k = 4
    # d = k - 1
    # model = GCNModel(k=k, d=d)
    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])
    # # valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, follow_batch=['x_src', 'x_dst'])
    # test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])
    #
    # gcn_model.train(train_loader, test_loader, model)
    # gcn_model.evaluation(valid_loader, model)
    # gcn_model.test(valid_loader, model)
