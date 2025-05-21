import marimo

__generated_with = "0.12.6"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import joblib
    import os
    return go, joblib, mo, nn, optim, os, pd, plotly, px, torch


@app.cell
def _():
    from pathlib import Path

    data_path=Path("data")

    genome_scores_data_path=data_path/"genome_scores.csv"
    genome_tags_data_path=data_path/"genome_tags.csv"
    link_data_path=data_path/"link.csv"
    movie_data_path=data_path / "movie.csv"
    rating_data_path= data_path / "rating.csv"
    tag_data_path= data_path / "tag.csv"
    return (
        Path,
        data_path,
        genome_scores_data_path,
        genome_tags_data_path,
        link_data_path,
        movie_data_path,
        rating_data_path,
        tag_data_path,
    )


@app.cell
def _(pd, rating_data_path):
    rating=pd.read_csv(rating_data_path)
    rating.timestamp=pd.to_datetime(rating.timestamp)
    rating["timestamp"]=pd.to_datetime(rating["timestamp"])
    rating:pd.DataFrame=rating.sort_values(by=["userId","timestamp"],ascending=[True,False])
    return (rating,)


@app.cell
def _(rating):
    rating
    return


@app.cell
def _(genome_scores_data_path, pd):
    genome_df=pd.read_csv(genome_scores_data_path)
    return (genome_df,)


@app.cell
def _(genome_df):
    genome_df
    return


@app.cell
def _(rating):
    print(len(rating.movieId.unique()))
    return


@app.cell
def _(rating):
    unique_movie_id=rating.movieId.unique()
    return (unique_movie_id,)


@app.cell
def _(genome_df):
    import numpy as np
    genome_group=genome_df.groupby("movieId")
    return genome_group, np


@app.cell
def _(genome_group, unique_movie_id):
    useful_genome_groups=[x for x in genome_group.groups if x in unique_movie_id]
    return (useful_genome_groups,)


@app.cell
def _(genome_group, np, useful_genome_groups):
    genome_matrix=np.array([np.array(genome_group.get_group(x)["relevance"].to_list()) for x in useful_genome_groups])
    return (genome_matrix,)


@app.cell
def _(genome_matrix):
    genome_matrix
    return


@app.cell
def _(nn):
    ## I wanna create en embedding model for the movies, since we have lot of infos for each movie, but i need to scale down a lot the dimensions, i think the best way is to use an autoencoder

    class Encoder(nn.Module):

        def __init__(self, n_labels, hidden_dimensions=512, n_layers=3, output_dim=10, dropout:float=0.2):
            super().__init__()
            layers=[]
            layers.append(nn.Linear(n_labels,hidden_dimensions))
            layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU())
            for _ in range(n_layers):
                layers.append(nn.Linear(hidden_dimensions,hidden_dimensions))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dimensions,output_dim))
            self.model=nn.Sequential(*layers)

        def forward(self,x):
            return self.model(x)

    class Decoder(nn.Module):

        def __init__(self, n_labels, hidden_dimensions=512, n_layers=3, output_dim=10,dropout:float=0.2):
            super().__init__()
            layers=[]
            layers.append(nn.Linear(output_dim,hidden_dimensions))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            for _ in range(n_layers):
                layers.append(nn.Linear(hidden_dimensions,hidden_dimensions))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dimensions,n_labels))
            self.model=nn.Sequential(*layers)

        def forward(self,x):
            return self.model(x)


    class AutoEncoder(nn.Module):

        def __init__(self, n_labels, hidden_dimensions=512, n_layers=3, output_dim=10,dropout:float=0.2):
            super().__init__()
            self.encoder=Encoder(n_labels, hidden_dimensions, n_layers, output_dim, dropout)
            self.decoder=Decoder(n_labels, hidden_dimensions, n_layers, output_dim, dropout)

        def forward(self,x):
            x=self.encode(x)
            x=self.decode(x)
            return x

        def encode(self,x):
            return self.encoder(x)

        def decode(self,x):
            return self.decoder(x)
    return AutoEncoder, Decoder, Encoder


@app.cell
def _(genome_matrix, pd, useful_genome_groups):
    genome_df_training=pd.DataFrame(genome_matrix,index=useful_genome_groups)
    return (genome_df_training,)


@app.cell
def _(genome_matrix, torch):
    import random
    from sklearn.model_selection import train_test_split


    train_sample,test_sample=train_test_split(genome_matrix,train_size=0.8)


    test_sample=torch.from_numpy(test_sample)
    return random, test_sample, train_sample, train_test_split


@app.cell
def _(AutoEncoder, genome_matrix, nn, optim, torch, train_sample):
    from torch.utils.data import DataLoader,Dataset



    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model=AutoEncoder(n_labels=len(genome_matrix[0]),hidden_dimensions=2048).to(device)
    dataloader=DataLoader(dataset=torch.from_numpy(train_sample),shuffle=True,batch_size=256)
    optimizer=optim.AdamW(params=model.parameters(),lr=0.0006)
    criterion=nn.MSELoss()
    return DataLoader, Dataset, criterion, dataloader, device, model, optimizer


@app.cell
def _(device, nn, torch):
    def save_encoder_model(model:nn.Module):
        torch.save(model.state_dict(),"models/encoder_model.pth")
    def validate_model(model,criterion,X,y=None,threshold=0.003,verbose=False):
        model.eval()
        with torch.no_grad():
            output=model(X.float().to(device))
            y=y or X
            y=y.float().to(device)
            loss=criterion(y,output)
            if verbose: print(f"Validation Loss: {loss.item()}")
            return loss.item()<=threshold,loss.item()
    return save_encoder_model, validate_model


@app.cell
def _(device, test_sample, torch, validate_model):
    import tqdm


    def train_encoder_model(model,dataloader,criterion,optimizer):
        losses=[]
        val_losses=[]
        for _ in tqdm.tqdm(range(500)):
            model.train()
            for batch in dataloader:
                batch=batch.float().to(device)
                with torch.autocast(device_type=device):
                    output=model(batch)
                    loss=criterion(batch,output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if _%10==0:
                is_model_validated,val_loss=validate_model(model,criterion=criterion,verbose=False,X=test_sample)
                val_losses.append(val_loss)
                if is_model_validated:
                    break
        return losses,val_losses
    return tqdm, train_encoder_model


@app.cell
def _(
    criterion,
    dataloader,
    model,
    optimizer,
    os,
    save_encoder_model,
    torch,
    train_encoder_model,
):
    if not os.path.exists("models/encoder_model.pth"):
        losses,val_losses=train_encoder_model(model,dataloader,criterion,optimizer)
        save_encoder_model(model)
    else:
        model.load_state_dict(torch.load("models/encoder_model.pth"))
    return losses, val_losses


@app.cell
def _(device, genome_matrix, model, torch):
    movie_matrix=torch.from_numpy(genome_matrix).float().to(device)
    movie_matrix_encoded=model.encode(movie_matrix)
    return movie_matrix, movie_matrix_encoded


@app.cell
def _(movie_matrix_encoded, pd, useful_genome_groups):
    movie_matrix_encoded_df=pd.DataFrame(movie_matrix_encoded.detach().cpu())
    movie_matrix_encoded_df.index=useful_genome_groups
    return (movie_matrix_encoded_df,)


@app.cell
def _():
    return


@app.cell
def _(genome_matrix, np, torch):
    torch.from_numpy(np.zeros(len(genome_matrix))).float()
    return


@app.cell
def _(device, genome_matrix, model, np, torch):
    model.encode(torch.from_numpy(np.zeros(len(genome_matrix[0]))).float().to(device))
    return


@app.cell
def _(movie_matrix_encoded_df):
    movie_matrix_encoded_df.hist(backend="plotly")
    return


@app.cell
def _(movie_matrix_encoded_df):
    movie_matrix_encoded_df.loc[1,:].values
    return


@app.cell
def _(rating):
    rating.drop_duplicates(subset=["userId","timestamp"],inplace=True)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Now we have to create a useful dataset to train a Neural Collaborative Filtering.
        To do that we should first understand what we need to teach to the model.

        1. We need a list of movies that a person has seen, is very important to order them by the time, since otherwise we would create a look ahead bias
        2. We need to insert some movies that the person has not seen, otherwise the model will learn only the movies that somebody likes
        """
    )
    return


@app.cell
def _(
    Dataset,
    data_path,
    device,
    genome_matrix,
    json,
    model,
    np,
    random,
    torch,
    tqdm,
):
    class MovieDataset(Dataset):
        def __init__(self, movie_ids, rating, movie_info, n_negatives=4, force_new_dataset=False, path="dataset_2",encode=False):
            self.path = path
            self.save_dir = data_path / self.path
            self.encoded_movies=encode
            if self.dataset_exists() and not force_new_dataset:
                self.load_dataset_torch()
            else:
                self.movie_not_seen_by_the_user = None
                self.movie, self.movie_infos, self.users, self.ratings, self.labels = self.build_dataset(
                     movie_ids, rating, movie_info, n_negatives
                )
                #self.save_dataset_torch()

        def dataset_exists(self):
            required_files = [
                "movie.pth", "movie_infos.pth", "users.pth",
                "ratings.pth", "labels.pth", "movie_not_seen_by_the_user.json"
            ]
            return all((self.save_dir / fname).exists() for fname in required_files)

        def save_dataset_torch(self):
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.movie, self.save_dir / "movie.pth")
            torch.save(self.movie_infos, self.save_dir / "movie_infos.pth")
            torch.save(self.users, self.save_dir / "users.pth")
            torch.save(self.ratings, self.save_dir / "ratings.pth")
            torch.save(self.labels, self.save_dir / "labels.pth")

            serializable_dict = {str(k): list(v) for k, v in self.movie_not_seen_by_the_user.items()}
            with open(self.save_dir / "movie_not_seen_by_the_user.json", "w") as f:
                json.dump(serializable_dict, f)

        def load_dataset_torch(self):
            self.movie = torch.load(self.save_dir / "movie.pt")
            self.movie_infos = torch.load(self.save_dir / "movie_infos.pt")
            self.users = torch.load(self.save_dir / "users.pt")
            self.ratings = torch.load(self.save_dir / "ratings.pt")
            self.labels = torch.load(self.save_dir / "labels.pt")

            with open(self.save_dir / "movie_not_seen_by_the_user.json", "r") as f:
                movie_dict = json.load(f)
            self.movie_not_seen_by_the_user = {int(k): v for k, v in movie_dict.items()}

        def __getitem__(self, idx):
            return (
                self.movie[idx],
                self.movie_infos[idx],
                self.users[idx],
                self.ratings[idx],
                self.labels[idx]
            )

        def __len__(self):
            return len(self.labels)

        def build_dataset(self, movie_ids, rating, movie_info, n_negatives=1):
            movies, movie_ids, users, ratings, labels = self.get_dataset(movie_ids, rating, movie_info, n_negatives)
            movies = self.convert_datas(movies, convert_to_float=True)
            movie_ids = self.convert_datas(movie_ids)
            users = self.convert_datas(users)
            ratings = self.convert_datas(ratings, convert_to_float=True)
            labels = self.convert_datas(labels)
            return movies, movie_ids, users, ratings, labels

        def get_constant(self):
            if self.encoded_movies:
                input_tensor = torch.from_numpy(np.zeros(len(genome_matrix[0]))).float().to(device)
                output = model.encode(input_tensor).detach().cpu().numpy()
            else:
                output = torch.from_numpy(np.zeros(len(genome_matrix[0]))).float()
            return output

        def get_dataset(self, all_movie_ids, rating, movie_info, n_negatives):
            movies, movie_ids, ratings_list, users, labels = [], [], [], [], []
            constant = self.get_constant()
            movie_not_seen_by_user = self.get_movies_not_seen_by_users(rating)
            training_dataset = self.get_training_dataset(rating)

            movie_users = set(zip(training_dataset["movieId"], training_dataset["userId"]))
            movie_users_rating = set(zip(movie_users, training_dataset["rating"]))

            for (movie_id, user_id), rating_val in tqdm.tqdm(movie_users_rating, desc="Building dataset"):
                movie_ids.append(movie_id)
                try:
                    movies.append(movie_info.loc[movie_id, :].values)
                except KeyError:
                    movies.append(constant)
                users.append(user_id)
                ratings_list.append(rating_val)
                labels.append(1)

                movies_not_seen = movie_not_seen_by_user.get(user_id, [])
                negative_samples = random.sample(movies_not_seen, k=min(n_negatives, len(movies_not_seen)))

                for negative_movie in negative_samples:
                    movie_ids.append(negative_movie)
                    if negative_movie in movie_info.index:
                        movies.append(movie_info.loc[negative_movie, :].values)
                    else:
                        movies.append(constant)
                    ratings_list.append(-1)
                    users.append(user_id)
                    labels.append(0)
            for movie in movies:
                if len(movie)!=1128:

                    print(movie.shape)
            return np.array(movies), np.array(movie_ids), np.array(users), np.array(ratings_list), np.array(labels)

        def get_training_dataset(self, rating):
            return rating.loc[rating.rank_latest > 1]

        def get_movies_not_seen_by_users(self, rating):
            if self.movie_not_seen_by_the_user is None:
                tmp_rating = rating[['userId', 'movieId']]
                tmp_rating.index = tmp_rating['movieId']
                movie_seen_by_users = tmp_rating.groupby("userId").groups
                total_movies = set(tmp_rating['movieId'].unique())
                self.movie_not_seen_by_the_user = {
                    user: list(total_movies - set(movie_ids))
                    for user, movie_ids in tqdm.tqdm(movie_seen_by_users.items(), desc="Calculating movies not seen by user")
                }
            return self.movie_not_seen_by_the_user

        @staticmethod
        def convert_datas(array, convert_to_float=False):
            tensor = torch.from_numpy(array)
            if convert_to_float:
                tensor = tensor.float()
            return tensor
    return (MovieDataset,)


@app.cell
def _(rating):
    movie_ids=rating.movieId.unique()
    return (movie_ids,)


@app.cell
def _(rating):
    rating.sort_values(by=["userId","timestamp"],ascending=[True,False],inplace=True)
    return


@app.cell
def _(rating):
    rating['rank_latest'] = rating.groupby(['userId'])['timestamp'] \
                                    .rank(method='first', ascending=False)
    return


@app.cell
def _(rating):
    test_ratings=rating.loc[rating.rank_latest<=3]
    train_ratings=rating.loc[rating.rank_latest>3]
    return test_ratings, train_ratings


@app.cell
def _(MovieDataset, genome_df_training, movie_ids, rating):
    train_dataset=MovieDataset(movie_ids,rating,genome_df_training)
    return (train_dataset,)


@app.cell
def _(nn, torch):
    class MovieEmbedder(nn.Module):

        def __init__(self,n_movies,input_features_dim,rating_dim,dim_embedding=10,multiplier=10):
            super().__init__()
            self.embedding=nn.Embedding(n_movies,embedding_dim=10)
            self.linear_layers=nn.Sequential(
                nn.Linear(1128+input_features_dim+rating_dim,512*multiplier),
                nn.ReLU(),
                nn.Linear(512*multiplier,256*multiplier),
                nn.ReLU(),
                nn.Linear(256*multiplier,128*multiplier),
                nn.ReLU(),
                nn.Linear(128*multiplier,128*multiplier),
                nn.ReLU(),
                nn.Linear(128*multiplier,dim_embedding),
                nn.ReLU()
            )

        def forward(self,x,features,rating):
            x=self.embedding(x)
            x=torch.cat([x,features,rating],dim=1)
            x=self.linear_layers(x)
            return x

        def embed(self,x):
            x=self.embedding(x)
            return x

    class MRS(nn.Module):
        def __init__(self,n_users,n_movies,input_features_dim,dim_movie_embedding=10,dim_users_embedding=10,rating_dim=1,multiplier=1):
            super().__init__()
            self.movie_embedder=MovieEmbedder(n_movies,input_features_dim,rating_dim,dim_movie_embedding)
            self.user_embedding=nn.Embedding(n_users,dim_users_embedding)
            self.linear_layers=nn.Sequential(
                nn.Linear(dim_movie_embedding+dim_users_embedding,1024*multiplier),
                nn.ReLU(),
                nn.Linear(1024*multiplier,512*multiplier),
                nn.ReLU(),
                nn.Linear(512*multiplier,256*multiplier),
                nn.ReLU(),
                nn.Linear(256*multiplier,128*multiplier),
                nn.ReLU(),
                nn.Linear(128*multiplier,64*multiplier),
                nn.ReLU(),
                nn.Linear(64*multiplier,32*multiplier),
                nn.ReLU(),
                nn.Linear(32*multiplier,1),
                nn.Sigmoid()
            )

            self.init_weights()

        def init_weights(self):
            def init_layer(layer):
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight) 
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            for layer in self.linear_layers:
                init_layer(layer=layer)

        def forward(self,movie,rating,user,movie_feature):
            movie=self.movie_embedder(movie,movie_feature,rating)
            users=self.user_embedding(user)
            x=torch.concat([movie,users],dim=1)
            x=self.linear_layers(x)
            return x
    return MRS, MovieEmbedder


@app.cell
def _(rating):
    rating
    return


@app.cell
def _(DataLoader, MRS, device, nn, optim, rating, train_dataset):
    train_data_loader=DataLoader(train_dataset,batch_size=512*4,shuffle=True)
    n_users=rating.userId.max()+1
    n_movies=rating.movieId.max()+1
    mrs=MRS(n_users=n_users,n_movies=n_movies,input_features_dim=10,dim_movie_embedding=10,dim_users_embedding=10,multiplier=4).to(device)
    optimizer_mrs=optim.Adam(mrs.parameters(),lr=0.0001)
    loss1=nn.BCELoss()
    return loss1, mrs, n_movies, n_users, optimizer_mrs, train_data_loader


@app.cell
def _(model):
    for parameter in model.parameters():
        print(len(parameter))
    return (parameter,)


@app.cell
def _(MRS, n_movies, n_users, torch, tqdm):
    n_epochs=2

    def save_model(model,path="models/model_final.pth"):
        torch.save(model.state_dict(),path)

    def train_model(n_epochs, train_data_loader, mrs, loss1,optimizer_mrs, device):
        from torch import autocast

        losses_1 = []

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            for movie_info, movie, user, rat, label in tqdm.tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}"):
                movie = movie.int().to(device)
                user = user.int().to(device)
                movie_info = movie_info.to(device)
                rat = rat.to(device).view(-1,1)
                label = label.float().to(device)

                watched= mrs(movie,rat, user, movie_info)
                watched = watched.view(-1)

                with autocast(device_type=device,dtype=torch.float16):
                    loss_1 = loss1(watched, label)


                optimizer_mrs.zero_grad()
                loss_1.backward()
                optimizer_mrs.step()

                losses_1.append(loss_1.item())

        return losses_1
    def load_model(path="models/model_final.pth",):
        model=MRS(n_users=n_users,
                  n_movies=n_movies,
                  input_features_dim=10,
                  dim_movie_embedding=10,
                  dim_users_embedding=10,
                  multiplier=4)
        model.load_state_dict(torch.load(path))
        return model
    return load_model, n_epochs, save_model, train_model


@app.cell
def _(device, loss1, mrs, optimizer_mrs, train_data_loader, train_model):
    losses_1=train_model(2, train_data_loader, mrs, loss1, optimizer_mrs, device)
    return (losses_1,)


@app.cell
def _(losses_1, px):
    loss_fig=px.line(losses_1)
    loss_fig.show()
    return (loss_fig,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Now we need to evaluate the model
        to do so we are gonna take 99 of the movies not seen by the user and 1 seen, we are gonna run the model on the 100 movies and ranked them based on the prob, if the movie is in the top 10 we say is a hit, otherwise it's not
        """
    )
    return


@app.cell
def _(rating, train_dataset):
    movies_not_seen_by_users=train_dataset.get_movies_not_seen_by_users(rating)
    return (movies_not_seen_by_users,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
