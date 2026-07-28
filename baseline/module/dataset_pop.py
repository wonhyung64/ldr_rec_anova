import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class UserItemTime(Dataset):
    def __init__(self, data_path, dataset, time_unit, time_len, seq_len):
        path = f"{data_path}/{dataset}"
        self.time_unit = time_unit

        self.time_dict = np.load(f'{path}/interaction_time_dict.npy', allow_pickle=True).item()
        self.train_dict = np.load(f'{path}/training_dict.npy', allow_pickle=True).item()
        self.valid_dict = np.load(f'{path}/validation_dict.npy', allow_pickle=True).item()
        self.test_dict = np.load(f'{path}/testing_dict.npy', allow_pickle=True).item()
        self.test_head_overall_dict = np.load(f'{path}/testing_dict_head_overall.npy', allow_pickle=True).item()
        self.test_head_recent_3d_dict = np.load(f'{path}/testing_dict_head_recent_3d.npy', allow_pickle=True).item()
        self.test_head_recent_7d_dict = np.load(f'{path}/testing_dict_head_recent_7d.npy', allow_pickle=True).item()
        self.test_tail_overall_dict = np.load(f'{path}/testing_dict_tail_overall.npy', allow_pickle=True).item()
        self.test_tail_recent_3d_dict = np.load(f'{path}/testing_dict_tail_recent_3d.npy', allow_pickle=True).item()
        self.test_tail_recent_7d_dict = np.load(f'{path}/testing_dict_tail_recent_7d.npy', allow_pickle=True).item()

        self.trainUniqueUsers, self.trainUser, self.trainItem, self.trainDataSize = self.load_set(self.train_dict)
        self.validUniqueUsers, self.validUser, self.validItem, self.validDataSize = self.load_set(self.valid_dict)
        self.testUniqueUsers, self.testUser, self.testItem, self.testDataSize = self.load_set(self.test_dict)

        self.m_item = max(self.trainItem.max(), self.validItem.max(), self.testItem.max()) + 1
        self.n_user = max(self.trainUser.max(), self.validUser.max(), self.testUser.max()) + 1

        self.UserItemNet = csr_matrix(
            (np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )

        self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)

        self.train_user_item_time = self.set_to_pair(self.train_dict, self.time_dict, self.time_unit)
        self.valid_user_item_time = self.set_to_pair(self.valid_dict, self.time_dict, self.time_unit)
        self.test_user_item_time = self.set_to_pair(self.test_dict, self.time_dict, self.time_unit)
        self.item_time_array = self.time_dict_to_array(self.time_dict, time_len)
        self.user_interactions = self.build_user_interactions(self.time_dict, self.time_unit)
        self.split_train_hot_n_cold()
        self.train_hist_item_list, self.train_hist_time_list = self.build_histories(self.train_hot_events, seq_len)


    def load_set(self, set_dict):
        UniqueUsers, Item, User = [], [], []
        dataSize = 0
        for uid in set_dict.keys():
            if len(set_dict[uid]) != 0:
                UniqueUsers.append(uid)
                User.extend([uid] * len(set_dict[uid]))
                Item.extend(set_dict[uid])
                dataSize += len(set_dict[uid])
        return np.array(UniqueUsers), np.array(User), np.array(Item), dataSize


    def set_to_pair(self, set_dict:dict, time_dict:dict, time_unit:str="d") -> dict:
        """Integrate splitted dataset and time.

        Args:
            set_dict (dict): Splitted dataset.
            time_dict (dict): Observed time for all interactions.
            time_unit (str, optional): Converting criteria of time from timestamp. Defaults to "d".

        Returns:
            dict: {(user_idx, item_idx): time}
        """
        user_item_time = {}
        for user in set_dict:
            for item in set_dict[user]:
                time = time_dict[user][item] 
                if time_unit == "s":
                    pass
                elif time_unit == "m":
                    time = time / 60
                elif time_unit == "h":
                    time = time / 60 / 60
                elif time_unit == "d":
                    time = time / 60 / 60 / 24
                user_item_time[(user, item)] = time
        return user_item_time


    def getUserPosItems(self, users, UserItemNet):
        posItems = []
        for user in users:
            posItems.append(UserItemNet[user, :].nonzero()[1])
        return posItems


    def time_dict_to_array(self, time_dict:dict, time_len:int=0) -> dict:
        """Observed times to numpy array.

        Args:
            time_dict (dict): Observed times
            time_len (int, optional): Number of recent time. Defaults to 0.

        Returns:
            numpy.ndarray: Interaction times for each item.
        """
        item_time_dict = {}
        for _, user_dict in time_dict.items():
            for item_idx, times in user_dict.items():
                try:
                    assert item_time_dict[item_idx]
                except:
                    item_time_dict[item_idx] = []
                item_time_dict[item_idx].append(times)

        item_time_array = []
        max_time = 0.
        max_time_len = 0
        for i in range(max(item_time_dict.keys())+1):
            try:
                times = np.array(item_time_dict[i])
                max_time = max(np.max(times), max_time)
            except:
                times = np.array([])
            times.sort()
            item_time_array.append(times)
            max_time_len = max(max_time_len, len(times))

        if time_len == 0:
            time_len = max_time_len

        for i in range(max(item_time_dict.keys())+1):
            time_array = item_time_array[i]
            item_time_array[i] = np.pad(time_array[-time_len:], (0, time_len - len(time_array[-time_len:])), "constant", constant_values=max_time)
            
        item_time_array = np.stack(item_time_array, 0)

        return item_time_array


    def build_user_interactions(self, time_dict:dict, time_unit:str) -> dict:
        """{user: {item: timestamp}} to {user: [(norm_time, item)]}

        Returns:
            dict
        """
        user_interactions = {u: [] for u in range(self.n_user)}
        for u, item_time in time_dict.items():
            for item, t in item_time.items():
                if time_unit == "s":
                    t = float(t)
                elif time_unit == "m":
                    t = float(t) / 60
                elif time_unit == "h":
                    t = float(t) / 60 / 60
                elif time_unit == "d":
                    t = float(t) / 60 / 60 / 24
                user_interactions[u].append((t, item))
        for u in user_interactions:
            user_interactions[u].sort(key=lambda x: x[0])
        return user_interactions


    def split_train_hot_n_cold(self):
        self.train_events = []
        for (u,v), t in self.train_user_item_time.items():
                self.train_events.append((u, v, float(t)))
        self.train_events.sort(key=lambda x: (x[0], x[2]))

        u_start = -1
        t_start = 0
        self.train_hot_events, self.train_cold_events = [], []
        for i, (u, v, t) in enumerate(self.train_events):
            if (u != u_start):
                self.train_cold_events.append((u, v, float(t)))
                u_start = u
                t_start = t 
            elif (u == u_start) & (t == t_start):
                self.train_cold_events.append((u, v, float(t)))
                t_start = t 
            elif (u == u_start) & (t != t_start):
                self.train_hot_events.append((u, v, float(t)))
                t_start = t 
            else:
                raise ValueError("Invalid Sample")

        self.train_hot_events.sort(key=lambda x: x[2])
        self.hot_user_list = np.array([u for (u, v, t) in self.train_hot_events], dtype=np.int64)
        self.hot_item_list = np.array([v for (u, v, t) in self.train_hot_events], dtype=np.int64)
        self.hot_event_time_list = np.array([t for (u, v, t) in self.train_hot_events], dtype=np.float32)
        self.hotDataSize = len(self.train_hot_events)

        self.train_cold_events.sort(key=lambda x: x[2])
        self.cold_user_list = np.array([u for (u, v, t) in self.train_cold_events], dtype=np.int64)
        self.cold_item_list = np.array([v for (u, v, t) in self.train_cold_events], dtype=np.int64)
        self.cold_event_time_list = np.array([t for (u, v, t) in self.train_cold_events], dtype=np.float32)
        self.coldDataSize = len(self.train_cold_events)

        self.trainDataSize = self.hotDataSize + self.coldDataSize


    def build_histories(self, events, seq_len):
        hist_item_list, hist_time_list = [], []
        for (u, v, t) in events:
            hist = [(tt, vv) for (tt, vv) in self.user_interactions[u] if tt < t]
            hist = hist[-seq_len:]
            h_items, h_times = [], []
            for (tt, vv) in hist:
                h_items.append(vv)
                h_times.append(tt)
            pad_len = seq_len - len(h_items)
            h_items = [self.m_item] * pad_len + h_items
            h_times = [0.] * pad_len + h_times
            hist_item_list.append(h_items)
            hist_time_list.append(h_times)
        return np.array(hist_item_list, dtype=np.int64), np.array(hist_time_list)


    def get_pair_user_uniform(self, k=1):
        pos_user = self.hot_user_list.astype(np.int64)
        N = len(pos_user)
        neg_user = np.random.randint(0, self.n_user - 1, size=(N, k), dtype=np.int64)
        neg_user += (neg_user >= pos_user[:, None])
        self.pos_user_list = pos_user
        self.neg_user_list = neg_user

    def get_pair_item_uniform(self, k=1, w_time=False):
        hot_pos_item = self.hot_item_list.astype(np.int64)
        hot_N = len(hot_pos_item)
        hot_neg_item = np.random.randint(0, self.m_item - 1, size=(hot_N, k), dtype=np.int64)
        hot_neg_item += (hot_neg_item >= hot_pos_item[:, None])
        self.hot_pos_item_list = hot_pos_item
        self.hot_neg_item_list = hot_neg_item

        cold_pos_item = self.cold_item_list.astype(np.int64)
        cold_N = len(cold_pos_item)
        cold_neg_item = np.random.randint(0, self.m_item - 1, size=(cold_N, k), dtype=np.int64)
        cold_neg_item += (cold_neg_item >= cold_pos_item[:, None])
        self.cold_pos_item_list = cold_pos_item
        self.cold_neg_item_list = cold_neg_item
        if w_time:
            self.hot_pos_time_all = self.item_time_array[hot_pos_item]
            self.hot_neg_time_all = self.item_time_array[hot_neg_item]
            self.cold_pos_time_all = self.item_time_array[cold_pos_item]
            self.cold_neg_time_all = self.item_time_array[cold_neg_item]
        


    def prepare_user_timebucket_sampler(self, bucket_size=86400, w_cold=False):
        self.user_bucket_size = float(bucket_size)

	    # active users per bucket from TRAIN events
        hot_bucket_to_users = {}
        for (u, v, t) in self.train_hot_events:
            b = int(t // self.user_bucket_size)
            if b not in hot_bucket_to_users:
                hot_bucket_to_users[b] = set()
            hot_bucket_to_users[b].add(u)

	    # store as numpy arrays for fast indexing
        self.hot_bucket_to_users_np = {
            b: np.array(sorted(list(users)), dtype=np.int64)
            for b, users in hot_bucket_to_users.items()
        }

	    # bucket id for each train event
        self.train_hot_event_bucket = np.array(
		    [int(t // self.user_bucket_size) for t in self.hot_event_time_list],
		    dtype=np.int64
	    )

	    # event indices grouped by bucket
        self.hot_bucket_to_event_idx = {}
        for idx, b in enumerate(self.train_hot_event_bucket):
            if b not in self.hot_bucket_to_event_idx:
                self.hot_bucket_to_event_idx[b] = []
            self.hot_bucket_to_event_idx[b].append(idx)
        self.hot_bucket_to_event_idx = {
            b: np.array(idxs, dtype=np.int64)
            for b, idxs in self.hot_bucket_to_event_idx.items()
        }

	    # global users for fallback
        self.hot_all_users_np = np.arange(self.n_user, dtype=np.int64)

        if w_cold:
            cold_bucket_to_users = {}
            for (u, v, t) in self.train_cold_events:
                b = int(t // self.user_bucket_size)
                if b not in cold_bucket_to_users:
                    cold_bucket_to_users[b] = set()
                cold_bucket_to_users[b].add(u)

            # store as numpy arrays for fast indexing
            self.cold_bucket_to_users_np = {
                b: np.array(sorted(list(users)), dtype=np.int64)
                for b, users in cold_bucket_to_users.items()
            }

            # bucket id for each train event
            self.train_cold_event_bucket = np.array(
                [int(t // self.user_bucket_size) for t in self.cold_event_time_list],
                dtype=np.int64
            )

            # event indices grouped by bucket
            self.cold_bucket_to_event_idx = {}
            for idx, b in enumerate(self.train_cold_event_bucket):
                if b not in self.cold_bucket_to_event_idx:
                    self.cold_bucket_to_event_idx[b] = []
                self.cold_bucket_to_event_idx[b].append(idx)
            self.cold_bucket_to_event_idx = {
                b: np.array(idxs, dtype=np.int64)
                for b, idxs in self.cold_bucket_to_event_idx.items()
            }

            # global users for fallback
            self.cold_all_users_np = np.arange(self.n_user, dtype=np.int64)

        
    def get_pair_user_event_timebucket_fast(self, k=1, w_cold=False):
        hot_pos_user = self.hot_user_list  # [N]
        N = len(hot_pos_user)

        hot_neg_user = np.empty((N, k), dtype=np.int64)

        for b, event_idx in self.hot_bucket_to_event_idx.items():
            users_arr = self.hot_bucket_to_users_np[b]   # active users in this bucket
            pos_u_b = hot_pos_user[event_idx]            # positive users for events in this bucket

            if len(users_arr) >= 2:
                # vectorized rejection sampling to avoid sampling the positive user
                rand_idx = np.random.randint(0, len(users_arr), size=len(event_idx))
                sampled = users_arr[rand_idx]

                mask = (sampled == pos_u_b)
                while mask.any():
                    rand_idx_resample = np.random.randint(0, len(users_arr), size=mask.sum())
                    sampled[mask] = users_arr[rand_idx_resample]
                    mask = (sampled == pos_u_b)

                hot_neg_user[event_idx, 0] = sampled

            else:
                u = pos_u_b
                r = np.random.randint(0, self.n_user - 1, size=len(event_idx))
                r += (r >= u)
                hot_neg_user[event_idx, 0] = r

        self.hot_neg_user_list = hot_neg_user.astype(np.int64)

        if w_cold:
            cold_pos_user = self.cold_user_list 
            M = len(cold_pos_user)

            cold_neg_user = np.empty((M, k), dtype=np.int64)

            for b, event_idx in self.cold_bucket_to_event_idx.items():
                users_arr = self.cold_bucket_to_users_np[b]   # active users in this bucket
                pos_u_b = cold_pos_user[event_idx]            # positive users for events in this bucket

                if len(users_arr) >= 2:
                    # vectorized rejection sampling to avoid sampling the positive user
                    rand_idx = np.random.randint(0, len(users_arr), size=len(event_idx))
                    sampled = users_arr[rand_idx]

                    mask = (sampled == pos_u_b)
                    while mask.any():
                        rand_idx_resample = np.random.randint(0, len(users_arr), size=mask.sum())
                        sampled[mask] = users_arr[rand_idx_resample]
                        mask = (sampled == pos_u_b)

                    cold_neg_user[event_idx, 0] = sampled

                else:
                    u = pos_u_b
                    r = np.random.randint(0, self.n_user - 1, size=len(event_idx))
                    r += (r >= u)
                    hot_neg_user[event_idx, 0] = r

            self.cold_neg_user_list = cold_neg_user.astype(np.int64)
