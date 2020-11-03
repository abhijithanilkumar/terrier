#pragma once

#include <deque>
#include <functional>
#include <iterator>
#include <stack>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "common/spin_latch.h"
#include "tbb/spin_rw_mutex.h"

namespace noisepage::storage::index {

/*
 * Structure of the B+ tree:
 *
 * The tree contains leaf nodes and inner nodes
 *
 * Inner node:
 *  Each inner node contains n-1 Keys and pointers, 1 previous pointer
 *
 *      Arrangement: |prev_ptr | K1, N1 | K2, N2 | ... | Kn-1, Nn-1|
 *      N1 contains keys in [K1, K2) and so on
 *      prev_ptr contains keys < K1
 *      Nn-1 contains keys >= Kn-1
 *
 * Leaf node:
 *  Each leaf node contains n - 1 keys and values, 1 previous pointer and 1 next pointer
 *  The next and previous pointers are used when we traverse the leaf nodes for scan ascending
 *  and scan descending.
 *
 *      Arrangement: |prev_ptr | K1, V1 | K2, V2 | ... | Kn-1, Vn-1 | next_ptr|
 *      Vi is a set that contains different values present for key Ki
 *      prev_ptr points to the previous leaf node with keys < K1
 *      next_ptr points to the next leaf node with keys > Kn-1
 *
 * Multi-threading:
 *  Each node (leaf and inner) contains one reader-writer lock (TBB) to gain access to the node.
 *  Writers are prioritized in this lock. Additionally, to guarentee safety of the root pointer of the
 *  tree we also have an additional latch for the root (root has 2 latches now). We acquire latches from
 *  top to bottom. Thus, before accessing a child node pointer, we are sure that the pointer cannot be
 *  deleted no matter what (lock is with us in parent).
 *
 *  We perform crab latching to acquire latches for reads and writes.
 *  Read:
 *    Acquire latches starting from root, in the path to find the key. Release read latch of parent
 *    once we get latch on current node and so on...
 *
 *  Write:
 *    Happens in 2 phases:
 *    1) Read latches acquired throughout the path expect in leaf node (write lock needed here). If the leaf node
 *    is safe for insertion/deletion perform the operation else we move to step 2.
 *
 *    2) Write latches are aquired from root to the corresponding leaf. We maintain a queue of parents on whom
 *    we hold latches. If the node we are at is safe, we release all parent locks.
 */

// Set all constants for the bplus tree nodes
#define FAN_OUT 200
// Ceil (FAN_OUT / 2) - 1
#define MIN_KEYS_INNER_NODE 99
#define MIN_PTR_INNER_NODE 100
// Ceil ((FAN_OUT - 1) / 2)
#define MIN_KEYS_LEAF_NODE 100

/*
 * BPlusTree - Implementation of a B+ Tree index
 *
 * Template Arguments:
 *
 * template <typename KeyType,
 *           typename ValueType,
 *           typename KeyComparator = std::less<KeyType>,
 *           typename KeyEqualityChecker = std::equal_to<KeyType>,
 *           typename KeyHashFunc = std::hash<KeyType>,
 *           typename ValueEqualityChecker = std::equal_to<ValueType>,
 *           typename ValueHashFunc = std::hash<ValueType>>
 *
 * Explanation:
 *
 *  - KeyType: Key type of the map
 *
 *  - ValueType: Value type of the map. Note that it is possible
 *               that a single key is mapped to multiple values
 *
 *  - KeyComparator: "less than" relation comparator for KeyType
 *                   Returns true if "less than" relation holds
 *                   *** NOTE: THIS OBJECT DO NOT NEED TO HAVE A DEFAULT
 *                   CONSTRUCTOR.
 *                   Please refer to main.cpp, class KeyComparator for more
 *                   information on how to define a proper key comparator
 *
 *  - KeyEqualityChecker: Equality checker for KeyType
 *                        Returns true if two keys are equal
 *
 *  - KeyHashFunc: Hashes KeyType into size_t. This is used in unordered_set
 *
 *  - ValueEqualityChecker: Equality checker for value type
 *                          Returns true for ValueTypes that are equal
 *
 *  - ValueHashFunc: Hashes ValueType into a size_t
 *                   This is used in unordered_set
 */
template <typename KeyType, typename ValueType, typename KeyComparator = std::less<KeyType>,
    typename KeyEqualityChecker = std::equal_to<KeyType>, typename KeyHashFunc = std::hash<KeyType>,
    typename ValueEqualityChecker = std::equal_to<ValueType>, typename ValueHashFunc = std::hash<ValueType>>
class BPlusTree {
  // static definition of comparators and equality checkers
  constexpr static const KeyComparator KEY_CMP_OBJ{};
  constexpr static const KeyEqualityChecker KEY_EQ_CHK{};
  constexpr static const ValueEqualityChecker VAL_EQ_CHK{};
  // We should protect the root ptr separately
  tbb::spin_rw_mutex root_latch_;

  /*
   * class Node - The base class for node types, i.e. InnerNode and LeafNode
   *
   * Since for InnerNode and LeafNode, the number of elements is not a compile
   * time known constant.
   */
  class Node {
   public:
    // Read-write latch present in the node
    tbb::spin_rw_mutex rw_latch_;

    /*
     * Constructor for the node (default one)
     */
    Node() = default;
    /*
     * Default destructor for the node
     */
    virtual ~Node() = default;

    /*
     * InnerNode and LeafNode inherit from node.
     *
     * These are the virtual functions which will be used by both the classes.
     * When we call one of these functions from a node, based on the type of the node,
     * (leaf or inner) the appropriate function is called
     */
    virtual Node *Split() = 0;
    virtual void SetPrevPtr(Node *ptr) = 0;
    virtual bool IsLeaf() = 0;
    virtual uint64_t GetSize() = 0;
    virtual size_t GetHeapSpaceSubtree() = 0;
    virtual Node *GetPrevPtr() = 0;
    virtual KeyType GetFirstKey() = 0;
    virtual KeyType GetLastKey() = 0;
    virtual void Append(Node *node) = 0;
    virtual bool WillOverflow() = 0;
    virtual bool WillUnderflow() = 0;
  };

  // Root of the tree
  Node *root_;

  // Datatypes for representing Node contents
  using ValueSet = std::unordered_set<ValueType, ValueHashFunc, ValueEqualityChecker>;
  using KeyValueSetPair = std::pair<KeyType, ValueSet>;
  using KeyNodePtrPair = std::pair<KeyType, Node *>;

  /*
   * LeafNode represents the leaf in the B+ Tree, storing the actual values in the index
   */
  class LeafNode : public Node {
   private:
    // Each key has a set of values stored as an unordered set
    std::vector<KeyValueSetPair> entries_;
    // Sibling pointers
    LeafNode *prev_ptr_;
    LeafNode *next_ptr_;

   public:
    /*
     * Constructor for the leaf node
     * Makes sure next and previous pointers are set accordingly
     */
    LeafNode() {
      prev_ptr_ = nullptr;
      next_ptr_ = nullptr;
    }

    /*
     * Destructor for the leaf node
     * Empties entries vector and sets previous and next pointers to null
     */
    ~LeafNode() override {
      entries_.clear();
      entries_.shrink_to_fit();
      prev_ptr_ = nullptr;
      next_ptr_ = nullptr;
    }

    /*
     * Inputs - key
     * Output - The index at which the givven key should be inserted at in the leaf node
     */
    int GetPositionToInsert(const KeyType &key) {
      ValueSet dummy_set;
      KeyValueSetPair entries_key = std::make_pair(key, dummy_set);

      auto it = std::lower_bound(entries_.begin(), entries_.end(), entries_key,
                                 [](const auto &a, const auto &b) { return KEY_CMP_OBJ(a.first, b.first); });

      return (it - entries_.begin());
    }

    /*
     * Inputs - key
     * Output - The highest index in the entries vector at which the entry's key is less than or equal
     * to the given key.
     */
    int GetPositionLessThanEqualTo(const KeyType &key) {
      ValueSet dummy_set;
      KeyValueSetPair entries_key = std::make_pair(key, dummy_set);

      auto it = std::upper_bound(entries_.begin(), entries_.end(), entries_key,
                                 [](const auto &a, const auto &b) { return KEY_CMP_OBJ(a.first, b.first); });

      return static_cast<int>(it - entries_.begin()) - 1;
    }

    /*
     * Inputs - key
     * Output - Returns an iterator for the entry with the given key
     */
    typename std::vector<KeyValueSetPair>::iterator GetPositionOfKey(const KeyType &key) {
      ValueSet dummy_set;
      KeyValueSetPair entries_key = std::make_pair(key, dummy_set);

      auto it = std::lower_bound(entries_.begin(), entries_.end(), entries_key,
                                 [](const auto &a, const auto &b) { return KEY_CMP_OBJ(a.first, b.first); });

      // Not guarenteed that the key exists in the entries
      if (it == entries_.end() || !KEY_EQ_CHK(it->first, key)) return entries_.end();

      return it;
    }

    /*
     * Returns the previous pointer for the node
     */
    Node *GetPrevPtr() override { return prev_ptr_; }

    /*
     * Check if the given node has overflown
     */
    bool IsOverflow() {
      uint64_t size = entries_.size();
      return (size >= FAN_OUT);
    }

    /*
     * Check if the given node has underflown
     */
    bool IsUnderflow() {
      uint64_t size = entries_.size();
      return (size < MIN_KEYS_LEAF_NODE);
    }

    /*
     * Check if the given node will underflow assuming deletion
     */
    bool WillUnderflow() override {
      uint64_t size = entries_.size();
      return ((size - 1) < MIN_KEYS_LEAF_NODE);
    }

    /*
     * Check if a node will overflow after an insertion
     */
    bool WillOverflow() override { return (entries_.size() == (FAN_OUT - 1)); }

    /*
     * Inputs - the pointer to the node which is to be pointed to by prev_ptr
     * Sets the previous pointer for the leaf node
     */
    void SetPrevPtr(Node *ptr) override { prev_ptr_ = dynamic_cast<LeafNode *>(ptr); }

    /*
     * Inputs - key
     * Output - Returns if the given key is present in the node or not
     */
    bool HasKey(const KeyType &key) {
      auto it = GetPositionOfKey(key);

      return !(it == entries_.end());
    }

    /*
     * Inputs - key, value
     * Output - Returns if the given key and value are present in the node or not
     */
    bool HasKeyValue(const KeyType &key, const ValueType &value) {
      auto it = GetPositionOfKey(key);

      if (it == entries_.end()) return false;

      auto loc = it->second.find(value);

      return (loc != it->second.end());
    }

    /*
     * Inputs - key, value
     * Inserts the given key and value into the node
     */
    void Insert(const KeyType &key, const ValueType &value) {
      uint64_t pos_to_insert = GetPositionToInsert(key);

      if (pos_to_insert < entries_.size() && KEY_EQ_CHK(entries_[pos_to_insert].first, key)) {
        entries_[pos_to_insert].second.insert(value);
      } else {
        KeyValueSetPair new_pair;
        new_pair.first = key;
        new_pair.second.insert(value);
        entries_.insert(entries_.begin() + pos_to_insert, new_pair);
      }
    }

    /*
     * Inputs - key, value_set
     * Inserts the given key and value_set into the node
     */
    void Insert(const KeyType &key, const ValueSet &value_set) {
      uint64_t pos_to_insert = GetPositionToInsert(key);

      if (pos_to_insert < entries_.size() && KEY_EQ_CHK(entries_[pos_to_insert].first, key)) {
        entries_[pos_to_insert].second = value_set;
      } else {
        KeyValueSetPair new_pair;
        new_pair.first = key;
        new_pair.second = value_set;
        entries_.insert(entries_.begin() + pos_to_insert, new_pair);
      }
    }

    /*
     * Returns the first key in the leaf node
     */
    KeyType GetFirstKey() override { return entries_[0].first; }

    /*
     * Returns the last key in the leaf node
     */
    KeyType GetLastKey() override { return entries_.rbegin()->first; }

    /*
     * Split the leaf node into two, returns the new node and sets sibling pointers
     */
    Node *Split() override {
      auto new_node = new LeafNode();

      // Copy the right half entries_ to the next node
      new_node->Copy(entries_.begin() + MIN_KEYS_LEAF_NODE, entries_.end());

      // Erase the right half from the current node
      entries_.erase(entries_.begin() + MIN_KEYS_LEAF_NODE, entries_.end());

      new_node->SetNextPtr(next_ptr_);
      if (next_ptr_) {
        next_ptr_->rw_latch_.lock();
        next_ptr_->SetPrevPtr(new_node);
        next_ptr_->rw_latch_.unlock();
      }

      // Set the forward sibling pointer of the current node
      next_ptr_ = new_node;

      // Set the backward sibling pointer of the new node
      new_node->SetPrevPtr(this);

      return new_node;
    }

    /*
     * Copy new entries into the leaf node
     */
    void Copy(typename std::vector<KeyValueSetPair>::iterator begin,
              typename std::vector<KeyValueSetPair>::iterator end) {
      entries_.insert(entries_.end(), begin, end);
    }

    /*
     * Inputs - key, predicate
     * Outputs - Returns if values in a key satisfies predicate (used for Conditional Insert)
     */
    bool SatisfiesPredicate(const KeyType &key, std::function<bool(const ValueType)> predicate) {
      auto it = GetPositionOfKey(key);

      if (it == entries_.end()) return false;

      for (auto i = (it->second).begin(); i != (it->second).end(); i++) {
        if (predicate(*i)) return true;
      }

      return false;
    }

    /*
     * Returns true, since this is a leaf node
     */
    bool IsLeaf() override { return true; }

    /*
     * Returns size (number of keys) in the node
     */
    uint64_t GetSize() override { return entries_.size(); }

    /*
     * Inputs - key, *results
     * Populate the values corresponding to a key into results vector
     */
    void ScanAndPopulateResults(const KeyType &key, typename std::vector<ValueType> *results) {
      auto it = GetPositionOfKey(key);

      if (it == entries_.end()) return;

      for (auto i = (it->second).begin(); i != (it->second).end(); i++) {
        results->push_back(*i);
      }
    }

    /*
     * Calculate the heap usage of the leaf node
     */
    size_t GetHeapSpaceSubtree() override {
      size_t size = 0;
      // Current node's heap space used
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        size += (it->second).size() * sizeof(ValueType) + sizeof(KeyType);
      }
      return size;
    }

    /*
     * Return the begin() iterator for entries_ in the node
     */
    typename std::vector<KeyValueSetPair>::iterator GetEntriesBegin() { return entries_.begin(); }

    /*
     * Return the end() iterator for entries_ in the node
     */
    typename std::vector<KeyValueSetPair>::iterator GetEntriesEnd() { return entries_.end(); }

    /*
     * Inputs - node
     * Append the entries in the node passed to the current node
     */
    void Append(Node *node) override {
      NOISEPAGE_ASSERT(node->IsLeaf(), "Node passed has to be a leaf.");
      auto node_ptr = dynamic_cast<LeafNode *>(node);
      entries_.insert(entries_.end(), node_ptr->GetEntriesBegin(), node_ptr->GetEntriesEnd());
    }

    /*
     * Remove the last (key, val) pair from the node and return it
     */
    KeyValueSetPair RemoveLastKeyValPair() {
      auto last_key_val_pair = *entries_.rbegin();
      entries_.erase(entries_.end() - 1);
      return last_key_val_pair;
    }

    /*
     * Remove the first (key, val) pair from the node and return it
     */
    KeyValueSetPair RemoveFirstKeyValPair() {
      auto first_key_val_pair = *entries_.begin();
      entries_.erase(entries_.begin());
      return first_key_val_pair;
    }

    /*
     * Inputs - key, value
     * Delete the corresponding (key, value) entry from the node
     */
    void DeleteEntry(const KeyType &key, const ValueType &value) {
      auto key_iter = GetPositionOfKey(key);
      auto value_iter = (key_iter->second).find(value);
      (key_iter->second).erase(value_iter);

      if ((key_iter->second).empty()) {
        entries_.erase(key_iter);
      }
    }

    /*
     * Returns the next pointer in a leaf node
     */
    LeafNode *GetNextPtr() { return next_ptr_; }

    /*
     * Inputs - node
     * Set the next pointer in a leaf node to node
     */
    void SetNextPtr(LeafNode *node) { next_ptr_ = node; }
  };

  /*
   * InnerNode represents one of the nodes in the non-leaf levels of the B+ Tree.
   */
  class InnerNode : public Node {
   private:
    // Node contains a vector of (key, ptr) pairs
    std::vector<KeyNodePtrPair> entries_;

    // n pointers, n-1 keys
    Node *prev_ptr_;

   public:
    /*
     * Constructor
     */
    InnerNode() { prev_ptr_ = nullptr; }

    /*
     * Destructor
     */
    ~InnerNode() override {
      entries_.clear();
      entries_.shrink_to_fit();
      prev_ptr_ = nullptr;
    }

    /*
     * Returns the prev_ptr
     */
    Node *GetPrevPtr() override { return prev_ptr_; }

    /*
     * Inputs - key
     * Output - Returns the first index whose key greater than or equal to the given key
     */
    int GetPositionGreaterThanEqualTo(const KeyType &key) {
      Node *dummy_node = nullptr;
      KeyNodePtrPair entries_key = std::make_pair(key, dummy_node);

      auto it = std::lower_bound(entries_.begin(), entries_.end(), entries_key,
                                 [](const auto &a, const auto &b) { return KEY_CMP_OBJ(a.first, b.first); });

      return (it - entries_.begin());
    }

    /*
     * Inputs - key
     * Returns the last index whose key less than or equal to the given key
     */
    int GetPositionLessThanEqualTo(const KeyType &key) {
      Node *dummy_node = nullptr;
      KeyNodePtrPair entries_key = std::make_pair(key, dummy_node);

      auto it = std::upper_bound(entries_.begin(), entries_.end(), entries_key,
                                 [](const auto &a, const auto &b) { return KEY_CMP_OBJ(a.first, b.first); });

      return static_cast<int>(it - entries_.begin()) - 1;
    }

    /*
     * Returns size (number of keys) in the node
     */
    uint64_t GetSize() override { return entries_.size(); }

    /*
     * Returns if the node has overflown
     * Assumption: prev_ptr_ is occupied, hence the check is >= FAN_OUT
     */
    bool IsOverflow() {
      uint64_t size = entries_.size();
      return (size >= FAN_OUT);
    }

    /*
     * Check if the given node has underflow
     */
    bool IsUnderflow() {
      // Function maybe called after deleting prev_ptr_
      uint64_t size = entries_.size() + (prev_ptr_ != nullptr);
      return (size < MIN_PTR_INNER_NODE);
    }

    /*
     * Check if the given node will underflow assuming deletion
     * Assumption: Used for borrowing, only looks at no. of keys
     */
    bool WillUnderflow() override {
      // Adding 1 assuming that prev_ptr_ is always occupied
      uint64_t size = entries_.size() + 1;
      return ((size - 1) < MIN_PTR_INNER_NODE);
    }

    /*
     * Check if the given node will overflow assuming insertion
     * Assumption: Used for borrowing, only looks at no. of keys
     */
    bool WillOverflow() override { return (entries_.size() == (FAN_OUT - 1)); }

    /*
     * Inputs - key, *locked_nodes
     * Acquires lock for predecessor and adds it to the locked_nodes queue
     * Output - Returns the predecessor of the node that has the given key
     */
    Node *GetPredecessor(const KeyType &key, std::deque<Node *> *locked_nodes) {
      int index = GetPositionLessThanEqualTo(key);

      // If your index is -1, you do not have a predecessor
      // Also means that the given key is pointed to by prev_ptr_
      if (index == -1) return nullptr;

      int pred_index = index - 1;

      // The predecessor is pointed to by prev_ptr_
      if (pred_index == -1) {
        // Get write lock
        prev_ptr_->rw_latch_.lock();
        locked_nodes->push_back(prev_ptr_);
        return prev_ptr_;
      }

      auto pred = entries_[pred_index].second;
      pred->rw_latch_.lock();
      locked_nodes->push_back(pred);
      return pred;
    }

    /*
     * Inputs - key, *locked_nodes
     * Acquires lock for successor and adds it to the locked_nodes queue
     * Output - Returns the successor of the node that has the given key
     */
    Node *GetSuccessor(const KeyType &key, std::deque<Node *> *locked_nodes) {
      int index = GetPositionLessThanEqualTo(key);

      uint64_t succ_index = index + 1;

      // The predecessor is pointed to by prev_ptr_
      if (succ_index == entries_.size()) return nullptr;

      auto successor = entries_[succ_index].second;
      successor->rw_latch_.lock();
      locked_nodes->push_back(successor);
      return successor;
    }

    /*
     * Return the begin() iterator for entries_ in the node
     */
    typename std::vector<KeyNodePtrPair>::iterator GetEntriesBegin() { return entries_.begin(); }

    /*
     * Return the end() iterator for entries_ in the node
     */
    typename std::vector<KeyNodePtrPair>::iterator GetEntriesEnd() { return entries_.end(); }

    /*
     * Inputs - node
     * Append the entries in the node passed to the current node
     */
    void Append(Node *node) override {
      NOISEPAGE_ASSERT(!node->IsLeaf(), "Node passed has to be an inner node.");
      auto node_ptr = dynamic_cast<InnerNode *>(node);
      entries_.insert(entries_.end(), node_ptr->GetEntriesBegin(), node_ptr->GetEntriesEnd());
    }

    /*
     * Inputs - key, node_ptr
     * Insert a new (key, node_ptr) pair into the node
     */
    void Insert(const KeyType &key, Node *node_ptr) {
      uint64_t pos_to_insert = GetPositionGreaterThanEqualTo(key);

      // We are sure that there is no duplicate key
      KeyNodePtrPair new_pair;
      new_pair.first = key;
      new_pair.second = node_ptr;
      entries_.insert(entries_.begin() + pos_to_insert, new_pair);
    }

    /*
     * Inputs - node
     * Set prev_ptr for the inner node (this will point to a node in the lower level)
     */
    void SetPrevPtr(Node *node) override { prev_ptr_ = node; }

    /*
     * Split the inner node and return the new node created
     */
    Node *Split() override {
      auto new_node = new InnerNode();

      // Copy the right half entries_ into the new node
      new_node->Copy(entries_.begin() + MIN_KEYS_INNER_NODE, entries_.end());

      // Delete the entries_ in the right half of the current node
      entries_.erase(entries_.begin() + MIN_KEYS_INNER_NODE, entries_.end());

      return new_node;
    }

    /*
     * Copy new entries into the node
     */
    void Copy(typename std::vector<KeyNodePtrPair>::iterator begin,
              typename std::vector<KeyNodePtrPair>::iterator end) {
      entries_.insert(entries_.end(), begin, end);
    }

    /*
     * Removes the first element in the node, used during insert overflow
     */
    KeyType RemoveFirstKey() {
      prev_ptr_ = entries_[0].second;
      KeyType first_key = entries_[0].first;
      entries_.erase(entries_.begin());
      return first_key;
    }

    /*
     * Returns false because this is an inner node
     */
    bool IsLeaf() override { return false; }

    /*
     * Inputs - key
     * Returns the (key, ptr) pair iterator corresponding to the key
     */
    typename std::vector<KeyNodePtrPair>::iterator GetKeyNodePtrPair(const KeyType &key) {
      int pos = GetPositionLessThanEqualTo(key);

      if (pos == -1) return entries_.end() - 1;

      return entries_.begin() + pos;
    }

    /*
     * Inputs - key
     * Returns the pointer corresponding to the key, used to traverse
     */
    Node *GetNodePtrForKey(const KeyType &key) {
      if (KEY_CMP_OBJ(key, (entries_.begin())->first)) {
        return prev_ptr_;
      }

      auto key_ptr_iter = GetKeyNodePtrPair(key);
      return key_ptr_iter->second;
    }

    /*
     * Return the space used by the subtree starting at this node
     */
    size_t GetHeapSpaceSubtree() override {
      size_t size = 0;

      NOISEPAGE_ASSERT(prev_ptr_ != nullptr, "There shouldn't be a node without prev ptr");
      // Space for subtree pointed to by previous
      size += prev_ptr_->GetHeapSpaceSubtree();

      // Current node's heap space used
      size += (entries_.capacity()) * sizeof(KeyNodePtrPair);

      // For all children
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        size += (it->second)->GetHeapSpaceSubtree();
      }

      return size;
    }

    /*
     * Get the first key in the node
     */
    KeyType GetFirstKey() override { return entries_[0].first; }

    /*
     * Get the last key in the node
     */
    KeyType GetLastKey() override { return entries_.rbegin()->first; }

    /*
     * Inputs - old_key, new_key
     * Replace the key that points to the old_key with new_key
     */
    KeyType ReplaceKey(const KeyType &old_key, const KeyType &new_key) {
      uint64_t key_pos = GetPositionLessThanEqualTo(old_key);
      KeyType old_parent_key = entries_[key_pos].first;
      entries_[key_pos].first = new_key;

      return old_parent_key;
    }

    /*
     * Remove the last (key, ptr) pair from the node and return it
     */
    KeyNodePtrPair RemoveLastKeyNodePtrPair() {
      auto last_key_ptr_pair = *entries_.rbegin();
      entries_.erase(entries_.end() - 1);
      return last_key_ptr_pair;
    }

    /*
     * Remove the first (key, ptr) pair from the node and return it
     */
    KeyNodePtrPair RemoveFirstKeyNodePtrPair() {
      auto first_key_ptr_pair = *entries_.begin();
      entries_.erase(entries_.begin());
      return first_key_ptr_pair;
    }

    /*
     * Delete the corresponding (key, value) entry from the node and return it
     */
    KeyType DeleteEntry(const KeyType &key) {
      auto key_iter = GetKeyNodePtrPair(key);
      KeyType deleted_key = key_iter->first;
      entries_.erase(key_iter);
      return deleted_key;
    }
  };

  /*
   * This class implements the iterator functionality used to traverse
   * the leaf nodes in the B+ tree.
   */
  class IndexIterator {
    // Node that the iterator is currently in
    LeafNode *current_;
    // Key offset at which the iterator is
    size_t key_offset_;
    // Value offset within a set for the given key (iterator is at this offset)
    size_t value_offset_;

   public:
    // Used to get the key of the iterator
    KeyType first_;
    // Used to get the value of the iterator
    ValueType second_;

    /*
     * Constructor
     */
    IndexIterator(LeafNode *c, size_t k, size_t v) {
      current_ = c;
      key_offset_ = k;
      value_offset_ = v;
      // Set first and second accordingly
      if (current_ != nullptr) {
        auto key_val_iter = (current_->GetEntriesBegin() + key_offset_);
        first_ = key_val_iter->first;
        second_ = *(std::next(key_val_iter->second.begin(), value_offset_));
      }
    }

    /*
     * Copy constructor
     */
    IndexIterator(const IndexIterator &itr) {
      current_ = itr.current_;
      key_offset_ = itr.key_offset_;
      value_offset_ = itr.value_offset_;
      first_ = itr.first_;
      second_ = itr.second_;
    }

    /*
     * Equality test for iterators
     */
    bool operator==(const IndexIterator &itr) {
      return (current_ == itr.current_ && key_offset_ == itr.key_offset_ && value_offset_ == itr.value_offset_);
    }

    /*
     * Moves the iterator to the next position (key, value)
     * Acts as the definition for the ++ operator
     */
    void operator++() {
      // ++ won't make you go outside block
      if (key_offset_ < current_->GetSize() - 1) {
        if (value_offset_ < (current_->GetEntriesBegin() + key_offset_)->second.size() - 1) {
          value_offset_++;
        } else {
          key_offset_++;
          value_offset_ = 0;
        }
      } else {
        // In last entry of block, if in last value as well
        if (value_offset_ < (current_->GetEntriesBegin() + key_offset_)->second.size() - 1) {
          value_offset_++;
        } else {
          // Have to move to the next leaf node
          NOISEPAGE_ASSERT(current_ != nullptr, "The ++ operator should not be called for a null iterator");
          auto new_current = current_->GetNextPtr();
          if (new_current != nullptr) {
            // Make scan retry if latch not acquired
            if (!new_current->rw_latch_.try_lock_read()) {
              current_->rw_latch_.unlock();
              current_ = nullptr;
              key_offset_ = 1;
              value_offset_ = 1;
              return;
            }
          }
          // Move to next node
          current_->rw_latch_.unlock();
          current_ = new_current;
          key_offset_ = 0;
          value_offset_ = 0;
        }
      }
      // Update the entry corresponding to the iterator
      if (current_ != nullptr) {
        auto key_val_iter = (current_->GetEntriesBegin() + key_offset_);
        first_ = key_val_iter->first;
        second_ = *(std::next(key_val_iter->second.begin(), value_offset_));
      }
    }

    /*
     * Moves the iterator to the previous position (key, value)
     * Acts as the definition for the -- operator
     */
    void operator--() {
      // If -- stays within current node
      if (key_offset_ > 0) {
        if (value_offset_ > 0) {
          value_offset_--;
        } else {
          key_offset_--;
          value_offset_ = (current_->GetEntriesBegin() + key_offset_)->second.size() - 1;
        }
      } else {
        // If -- stays within first key entry
        if (value_offset_ > 0) {
          value_offset_--;
        } else {
          // If we have to move one node before
          NOISEPAGE_ASSERT(current_ != nullptr, "The -- operator should not be called for a null iterator");
          auto new_current = dynamic_cast<LeafNode *>(current_->GetPrevPtr());
          if (new_current != nullptr) {
            // Make scan retry if latch not acquired
            if (!new_current->rw_latch_.try_lock_read()) {
              current_->rw_latch_.unlock();
              current_ = nullptr;
              key_offset_ = 1;
              value_offset_ = 1;
              return;
            }
            // Last key, value pair in the node
            key_offset_ = new_current->GetSize() - 1;
            value_offset_ = (new_current->GetEntriesBegin() + key_offset_)->second.size() - 1;
          } else {
            key_offset_ = 0;
            value_offset_ = 0;
          }
          // Move to previous node
          current_->rw_latch_.unlock();
          current_ = new_current;
        }
      }
      // Calculate the corresponding entry for the iterator
      if (current_ != nullptr) {
        auto key_val_iter = (current_->GetEntriesBegin() + key_offset_);
        first_ = key_val_iter->first;
        second_ = *(std::next(key_val_iter->second.begin(), value_offset_));
      }
    }

    /*
     * Unlocks latch on the current node
     * Used in limit scans
     */
    void Unlock() { current_->rw_latch_.unlock(); }
  };

  /*
   * Inputs - key, node_traceback, write_lock_leaf
   * Traverse and find the leaf node that has the given key, populate the stack to store the path.
   * Crab latching is done while reading from nodes.When function returns, leaf node with write
   * lock acquired is returned.
   * Output - The leaf node which contains the given key
   */
  LeafNode *FindLeafNode(const KeyType &key, std::stack<InnerNode *> *node_traceback, bool write_lock_leaf = false) {
    Node *node;

    // Spin to get the latch on the root (root might be updatesd over iterations)
    if (write_lock_leaf) {
      root_latch_.lock();
      root_->rw_latch_.lock();
    } else {
      root_latch_.lock_read();
      root_->rw_latch_.lock_read();
    }
    node = root_;

    while (!node->IsLeaf()) {
      auto inner_node = dynamic_cast<InnerNode *>(node);

      // Acquire read lock for the node
      if (inner_node != root_) {
        inner_node->rw_latch_.lock_read();
      }

      // If parent exists, release read lock
      if (!node_traceback->empty()) {
        auto parent = node_traceback->top();
        parent->rw_latch_.unlock();
        if (parent == root_) root_latch_.unlock();
      }

      // Add to stack to store path to found node
      node_traceback->push(inner_node);

      // Find the link to the next node based on the key
      node = inner_node->GetNodePtrForKey(key);
    }

    if (node != root_) {
      if (write_lock_leaf) {
        // Get write lock for leaf
        node->rw_latch_.lock();
      } else {
        node->rw_latch_.lock_read();
      }
    }

    // If parent exists, release read lock
    if (!node_traceback->empty()) {
      auto parent = node_traceback->top();
      parent->rw_latch_.unlock();
      if (parent == root_) root_latch_.unlock();
    }

    return dynamic_cast<LeafNode *>(node);
  }

  /*
   * Inputs - node, is_delete
   * Returns if the given node is safe depending on delete/insert (is_delete gives us this info)
   */
  bool IsSafe(Node *node, bool is_delete) {
    if (is_delete) return !node->WillUnderflow();
    return !node->WillOverflow();
  }

  /*
   * Inputs - key, node_traceback, locked_nodes, is_delete
   * Traverse and find the leaf node that has the given key, populate the stack to store the path.
   * Crab latching is performed for write latches throughout the path from the root. locked_nodes keeps
   * track of the nodes which are locked during crab latching. Safety of a node during crab latching is
   * determined accordingly based on is_delete (insert/delete).
   */
  LeafNode *FindLeafNodeWrite(const KeyType &key, std::stack<InnerNode *> *node_traceback,
                              std::deque<Node *> *locked_nodes, bool is_delete) {
    Node *node;

    // Acquire both root latches
    root_latch_.lock();
    root_->rw_latch_.lock();
    node = root_;

    while (!node->IsLeaf()) {
      auto inner_node = dynamic_cast<InnerNode *>(node);

      // Acquire read lock for the node
      if (inner_node != root_) {
        inner_node->rw_latch_.lock();
      }

      // If parent exists, release read lock
      if (!locked_nodes->empty() && IsSafe(node, is_delete)) {
        ReleaseNodeLocks(locked_nodes);
      }

      // After releasing all locks for parents
      locked_nodes->push_back(inner_node);

      // Add to stack to store path to found node
      node_traceback->push(inner_node);

      // Find the link to the next node based on the key
      node = inner_node->GetNodePtrForKey(key);
    }

    if (node != root_) {
      // Get write lock for leaf
      node->rw_latch_.lock();
    }

    // If parent exists, release locks
    if (!locked_nodes->empty() && IsSafe(node, is_delete)) {
      ReleaseNodeLocks(locked_nodes);
    }

    locked_nodes->push_back(node);

    return dynamic_cast<LeafNode *>(node);
  }

  /*
   * Inputs - key, value, insert_node, node_traceback
   * Insert a new (key, value) pair in the leaf node of the tree and propagate
   * changes up the tree.
   */
  void InsertAndPropagate(const KeyType &key, const ValueType &value, LeafNode *insert_node,
                          std::stack<InnerNode *> *node_traceback) {
    insert_node->Insert(key, value);
    // If insertion causes overflow
    if (insert_node->IsOverflow()) {
      // Split the insert node and return the new (right) node with all values updated
      auto child_node = dynamic_cast<LeafNode *>(insert_node->Split());

      // insert_node : left
      // child_node : right
      if (insert_node == root_) {
        auto new_root = new InnerNode();
        new_root->Insert(child_node->GetFirstKey(), child_node);
        new_root->SetPrevPtr(insert_node);
        root_ = dynamic_cast<Node *>(new_root);
        root_latch_.unlock();
        return;
      }

      auto parent_node = dynamic_cast<InnerNode *>(node_traceback->top());
      node_traceback->pop();

      // Insert the leaf node at the inner node and propagate
      InnerNode *current_node = parent_node;
      // Since child_node is always a leaf node, we do not remove first key here
      NOISEPAGE_ASSERT(child_node->IsLeaf(), "child_node has to be a leaf node");
      KeyType middle_key = child_node->GetFirstKey();
      Node *child = child_node;

      // If current node is the root or has a parent
      while (current_node == root_ || !node_traceback->empty()) {
        // Insert child node into the current node
        current_node->Insert(middle_key, child);

        // Overflow
        if (current_node->IsOverflow()) {
          Node *new_node = current_node->Split();
          middle_key = dynamic_cast<InnerNode *>(new_node)->RemoveFirstKey();

          // Context we have is the left node
          // new_node is the right node
          if (current_node == root_) {
            NOISEPAGE_ASSERT(node_traceback->empty(), "Stack should be empty when current node is the root");

            auto new_root = new InnerNode();
            // Will basically insert the key and node as node is empty
            new_root->Insert(middle_key, new_node);
            new_root->SetPrevPtr(current_node);
            root_ = new_root;
            root_latch_.unlock();
            return;
          }

          // Update current_node to the parent
          current_node = node_traceback->top();
          node_traceback->pop();
          child = new_node;
        } else {
          // Insertion does not cause overflow
          break;
        }
      }
    }
  }

  /*
   * Inputs - left_sibling, node, parent
   * Borrow an entry from left leaf node into node, and update the parent accordingly
   */
  void BorrowFromLeftLeaf(LeafNode *left_sibling, LeafNode *node, InnerNode *parent) {
    KeyValueSetPair last_key_val_pair = left_sibling->RemoveLastKeyValPair();

    // GetFirstKey() might not be present in the parent node, but we replace the corresponding key
    parent->ReplaceKey(node->GetFirstKey(), last_key_val_pair.first);

    node->Insert(last_key_val_pair.first, last_key_val_pair.second);
  }

  /*
   * Inputs - right_sibling, node, parent
   * Borrow an entry from right leaf node into node, and update the parent accordingly
   */
  void BorrowFromRightLeaf(LeafNode *right_sibling, LeafNode *node, InnerNode *parent) {
    KeyValueSetPair first_key_val_pair = right_sibling->RemoveFirstKeyValPair();

    // The key might not be present in the parent node, but we replace the corresponding key
    parent->ReplaceKey(first_key_val_pair.first, right_sibling->GetFirstKey());

    node->Insert(first_key_val_pair.first, first_key_val_pair.second);
  }

  /*
   * Inputs - left_sibling, node, parent
   * Borrow an entry from left inner node into node and update the parent accordingly
   */
  void BorrowFromLeftInner(InnerNode *left_sibling, InnerNode *node, InnerNode *parent) {
    KeyNodePtrPair last_key_node_pair = left_sibling->RemoveLastKeyNodePtrPair();

    // GetFirstKey() might not be present in the parent node, but we replace the corresponding key
    KeyType old_parent_key = parent->ReplaceKey(node->GetFirstKey(), last_key_node_pair.first);

    node->Insert(old_parent_key, node->GetPrevPtr());

    node->SetPrevPtr(last_key_node_pair.second);
  }

  /*
   * Inputs - right_sibling, node, parent
   * Borrow an entry from right inner node into node, and update the parent accordingly
   */
  void BorrowFromRightInner(InnerNode *right_sibling, InnerNode *node, InnerNode *parent) {
    // Remove first key value pair (key is transferred to parent, Node pointer becomes new prev pointer)
    KeyNodePtrPair first_key_node_pair = right_sibling->RemoveFirstKeyNodePtrPair();

    // Replace the key in the parent with the next lowest key in the right subtree
    KeyType old_parent_key = parent->ReplaceKey(first_key_node_pair.first, first_key_node_pair.first);

    node->Insert(old_parent_key, right_sibling->GetPrevPtr());
    right_sibling->SetPrevPtr(first_key_node_pair.second);
  }

  /*
   * Inputs - src, dst, parent
   * Coalesce from src to dst (right to left)
   */
  void CoalesceLeaf(LeafNode *src, LeafNode *dst, InnerNode *parent) {
    NOISEPAGE_ASSERT(src != nullptr, "source is non null");
    // Both src and dst of same level

    // Deletes the entry pointing to src node
    if (src == nullptr) return;
    parent->DeleteEntry(src->GetFirstKey());

    // Copy entries
    dst->Append(src);
  }

  /*
   * Inputs - src, dst, parent
   * Coalesce from source to destination (right to left)
   */
  void CoalesceInner(InnerNode *src, InnerNode *dst, InnerNode *parent) {
    NOISEPAGE_ASSERT(src != nullptr, "source is non null");
    // Both src and dst of same level
    // Deletes the entry pointing to src node
    if (src == nullptr) return;
    KeyType parent_key = parent->DeleteEntry(src->GetFirstKey());

    dst->Insert(parent_key, src->GetPrevPtr());

    // Copy entries
    dst->Append(src);
  }

  /*
   * Inputs - node, locked_nodes
   * Release lock for node and remove it from the locked_nodes queue
   */
  void RemoveFromLockList(Node *node, std::deque<Node *> *locked_nodes) {
    for (auto it = locked_nodes->begin(); it != locked_nodes->end(); ++it) {
      if (*it == node) {
        (*it)->rw_latch_.unlock();
        if (*it == root_) {
          root_latch_.unlock();
        }
        locked_nodes->erase(it);
        return;
      }
    }
  }

  /*
   * Inputs - locked_nodes
   * Checks if the root of the tree is present in the locked_nodes queue
   */
  bool IsRootPresent(std::deque<Node *> *locked_nodes) {
    for (auto it = locked_nodes->begin(); it != locked_nodes->end(); ++it) {
      if (*it == root_) {
        return true;
      }
    }
    return false;
  }

  /*
   * Inputs - node, node_traceback, locked_nodes
   * Balance a tree on deletion at leaf node (node) using the node_traceback stack and release locks from
   * locked_nodes as and when required
   */
  void Balance(LeafNode *node, std::stack<InnerNode *> *node_traceback, std::deque<Node *> *locked_nodes) {
    // Handle leaf merge separately
    auto parent_node = node_traceback->top();
    auto left_sibling = dynamic_cast<LeafNode *>(parent_node->GetPredecessor(node->GetFirstKey(), locked_nodes));
    auto right_sibling = dynamic_cast<LeafNode *>(parent_node->GetSuccessor(node->GetFirstKey(), locked_nodes));

    // Borrow from siblings if possible
    if (left_sibling && !left_sibling->WillUnderflow()) {
      BorrowFromLeftLeaf(left_sibling, node, parent_node);
      return;
    }
    if (right_sibling && !right_sibling->WillUnderflow()) {
      BorrowFromRightLeaf(right_sibling, node, parent_node);
      return;
    }

    // Try Coalesce
    // Parent node entry for child is also deleted
    if (left_sibling) {
      CoalesceLeaf(node, left_sibling, parent_node);
      left_sibling->SetNextPtr(node->GetNextPtr());
      if (node->GetNextPtr() != nullptr) {
        node->GetNextPtr()->SetPrevPtr(left_sibling);
      }
      // Remove from list of nodes with active lock
      RemoveFromLockList(node, locked_nodes);
      RemoveFromLockList(left_sibling, locked_nodes);
      delete node;
    } else if (right_sibling) {
      CoalesceLeaf(right_sibling, node, parent_node);
      node->SetNextPtr(right_sibling->GetNextPtr());
      if (right_sibling->GetNextPtr() != nullptr) {
        right_sibling->GetNextPtr()->SetPrevPtr(node);
      }
      RemoveFromLockList(right_sibling, locked_nodes);
      RemoveFromLockList(node, locked_nodes);
      delete right_sibling;
    } else {
      // Do Nothing
    }

    node_traceback->pop();

    // Handle inner nodes now
    auto inner_node = parent_node;

    while (inner_node == root_ || inner_node->IsUnderflow()) {
      if (inner_node == root_) {
        if (inner_node->GetSize() == 0) {
          // Only 1 pointer left in the root
          auto tmp = root_;
          root_ = root_->GetPrevPtr();
          // Check if the new root must be locked again
          if (!IsRootPresent(locked_nodes)) {
            root_->rw_latch_.lock();
            locked_nodes->push_back(root_);
          }
          RemoveFromLockList(tmp, locked_nodes);
          delete tmp;
        }
        return;
      }

      parent_node = node_traceback->top();
      node_traceback->pop();
      auto left_inner = dynamic_cast<InnerNode *>(parent_node->GetPredecessor(inner_node->GetFirstKey(), locked_nodes));
      auto right_inner = dynamic_cast<InnerNode *>(parent_node->GetSuccessor(inner_node->GetFirstKey(), locked_nodes));

      if ((left_inner != nullptr) && !left_inner->WillUnderflow()) {
        BorrowFromLeftInner(left_inner, inner_node, parent_node);
        return;
      }
      if ((right_inner != nullptr) && !right_inner->WillUnderflow()) {
        BorrowFromRightInner(right_inner, inner_node, parent_node);
        return;
      }

      // Try Coalesce
      // Parent node entry for child is also deleted
      if (left_inner != nullptr) {
        CoalesceInner(inner_node, left_inner, parent_node);
        // Remove from list of nodes with active lock
        RemoveFromLockList(inner_node, locked_nodes);
        delete inner_node;
      } else {
        CoalesceInner(right_inner, inner_node, parent_node);
        RemoveFromLockList(right_inner, locked_nodes);
        delete right_inner;
      }

      inner_node = parent_node;
    }
  }

 public:
  /*
   * Constructor for B+ tree
   * Root is never nullptr. Starts off as empty leaf node.
   */
  BPlusTree() { root_ = new LeafNode(); }

  void DeleteTree(Node *node) {
    if (node->GetSize() == 0) {
      delete node;
      return;
    }

    if (!node->IsLeaf()) {
      DeleteTree(node->GetPrevPtr());
      auto it = dynamic_cast<InnerNode *>(node)->GetEntriesBegin();
      for (size_t i = 0; i < node->GetSize(); i++) {
        DeleteTree((it + i)->second);
      }
    }

    delete node;
  }

  ~BPlusTree() { DeleteTree(root_); }

  /*
   * Returns the root of the B+ tree
   */
  Node *GetRoot() { return root_; }

  /*
   * Inputs - locked_nodes
   * Releases locks held by nodes in the locked nodes queue
   */
  void ReleaseNodeLocks(std::deque<Node *> *locked_nodes) {
    // Release all held locks
    while (!locked_nodes->empty()) {
      auto node = locked_nodes->front();
      locked_nodes->pop_front();
      node->rw_latch_.unlock();
      if (node == root_) root_latch_.unlock();
    }
  }

  /*
   * Inputs - key, value, unique_key
   * API to insert a new (key, value) pair into the tree
   */
  bool Insert(const KeyType &key, const ValueType &value, bool unique_key = false) {
    std::stack<InnerNode *> node_traceback;
    LeafNode *insert_node;

    insert_node = FindLeafNode(key, &node_traceback, true);  // Node traceback passed as ref

    // If there were conflicting key values
    if (insert_node->HasKeyValue(key, value) || (unique_key && insert_node->HasKey(key))) {
      insert_node->rw_latch_.unlock();
      if (insert_node == root_) root_latch_.unlock();
      return false;  // The traverse function aborts the insert as key, val is present
    }

    if (!insert_node->WillOverflow()) {
      InsertAndPropagate(key, value, insert_node, &node_traceback);
      insert_node->rw_latch_.unlock();
      if (insert_node == root_) root_latch_.unlock();
    } else {
      // Release write lock aquired while finding
      insert_node->rw_latch_.unlock();
      if (insert_node == root_) root_latch_.unlock();

      while (!node_traceback.empty()) {
        node_traceback.pop();
      }

      // A queue of nodes which contain write locks
      std::deque<Node *> locked_nodes;
      insert_node = FindLeafNodeWrite(key, &node_traceback, &locked_nodes, false);  // Node traceback passed as ref

      // If there were conflicting key values
      if (insert_node->HasKeyValue(key, value) || (unique_key && insert_node->HasKey(key))) {
        ReleaseNodeLocks(&locked_nodes);
        return false;  // The traverse function aborts the insert as key, val is present
      }

      InsertAndPropagate(key, value, insert_node, &node_traceback);

      ReleaseNodeLocks(&locked_nodes);
    }

    return true;
  }

  /*
   * Inputs - key, value, predicate, predicate_satisfied
   * API to perform (key, value) pair insert based on a predicate into the tree
   * predicate_satisfied is set based on predicate
   */
  bool ConditionalInsert(const KeyType &key, const ValueType &value, std::function<bool(const ValueType)> predicate,
                         bool *predicate_satisfied) {
    LeafNode *insert_node;
    std::stack<InnerNode *> node_traceback;

    insert_node = FindLeafNode(key, &node_traceback, true);  // Node traceback passed as ref

    // If there were conflicting key values
    if (insert_node->SatisfiesPredicate(key, predicate)) {
      insert_node->rw_latch_.unlock();
      if (insert_node == root_) root_latch_.unlock();
      *predicate_satisfied = true;
      return false;  // The traverse function aborts the insert as key, val is present
    }

    *predicate_satisfied = false;

    if (!insert_node->WillOverflow()) {
      InsertAndPropagate(key, value, insert_node, &node_traceback);
      insert_node->rw_latch_.unlock();
      if (insert_node == root_) root_latch_.unlock();
    } else {
      // Release write lock aquired
      insert_node->rw_latch_.unlock();
      if (insert_node == root_) root_latch_.unlock();

      // Redo the search by acquiring write locks

      while (!node_traceback.empty()) {
        node_traceback.pop();
      }

      // A queue of nodes which contain write locks
      std::deque<Node *> locked_nodes;
      insert_node = FindLeafNodeWrite(key, &node_traceback, &locked_nodes, false);  // Node traceback passed as ref

      // If there were conflicting key values
      if (insert_node->SatisfiesPredicate(key, predicate)) {
        ReleaseNodeLocks(&locked_nodes);
        *predicate_satisfied = true;
        return false;  // The traverse function aborts the insert as key, val is present
      }

      *predicate_satisfied = false;

      InsertAndPropagate(key, value, insert_node, &node_traceback);

      ReleaseNodeLocks(&locked_nodes);
    }

    return true;
  }

  /*
   * Inputs - key, results
   * API to fetch the values stored in the corresponding key and populate a vector with it
   */
  void GetValue(const KeyType &key, typename std::vector<ValueType> *results) {
    std::stack<InnerNode *> node_traceback;

    LeafNode *node = FindLeafNode(key, &node_traceback, false);

    node->ScanAndPopulateResults(key, results);

    // Release read lock
    node->rw_latch_.unlock();
    if (node == root_) root_latch_.unlock();
  }

  /*
   * API to calculate heap usage
   */
  size_t GetHeapUsage() {
    if (root_->GetSize() == 0) {
      return 0;
    }

    return root_->GetHeapSpaceSubtree();
  }

  /*
   * API to get the height of the tree
   */
  size_t GetHeightOfTree() {
    size_t height = 1;

    Node *node = root_;

    if (node->GetSize() == 0) return 0;

    while (!node->IsLeaf()) {
      height++;
      node = node->GetPrevPtr();
    }

    return height;
  }

  /*
   * API to delete an entry in the tree
   */
  bool Delete(const KeyType &key, const ValueType &value) {
    std::stack<InnerNode *> node_traceback;
    auto node = FindLeafNode(key, &node_traceback, true);

    if (!node->HasKeyValue(key, value)) {
      node->rw_latch_.unlock();
      if (node == root_) root_latch_.unlock();
      return false;
    }

    if (node == root_) {
      node->DeleteEntry(key, value);
      node->rw_latch_.unlock();
      root_latch_.unlock();
      // Do nothing as we allow the root to have 0 entries when it is a leaf node
      return true;
    }

    if (!node->WillUnderflow()) {
      // Delete and return
      node->DeleteEntry(key, value);
      node->rw_latch_.unlock();
      if (node == root_) root_latch_.unlock();
    } else {
      // Must propagate changes up
      // Release the lock
      node->rw_latch_.unlock();
      if (node == root_) root_latch_.unlock();

      while (!node_traceback.empty()) {
        node_traceback.pop();
      }

      std::deque<Node *> locked_nodes;

      node = FindLeafNodeWrite(key, &node_traceback, &locked_nodes, true);

      // Delete entry for leaf node
      node->DeleteEntry(key, value);

      if (node == root_) {
        ReleaseNodeLocks(&locked_nodes);
        // Do nothing as we allow the root to have 0 entries when it is a leaf node
        return true;
      }

      // Must propagate
      if (node->IsUnderflow()) {
        // Balance at leaf level
        Balance(node, &node_traceback, &locked_nodes);
      }

      ReleaseNodeLocks(&locked_nodes);
    }

    return true;
  }

  /*
   * Returns the first iterator in the tree
   */
  IndexIterator Begin() {
    root_latch_.lock_read();
    root_->rw_latch_.lock_read();
    Node *node = root_;

    // If root is empty
    if (node->GetSize() == 0) {
      root_->rw_latch_.unlock();
      root_latch_.unlock();
      return End();
    }

    // Find leaf node
    while (!node->IsLeaf()) {
      auto child = node->GetPrevPtr();
      child->rw_latch_.lock_read();
      node->rw_latch_.unlock();
      if (node == root_) root_latch_.unlock();
      node = child;
    }

    // Hold root's latch but release root_latch_
    if (node == root_) root_latch_.unlock();

    return IndexIterator(dynamic_cast<LeafNode *>(node), 0, 0);
  }

  /*
   * Inputs - key
   * Returns the first iterator which has key >= key
   */
  IndexIterator Begin(const KeyType &key) {
    std::stack<InnerNode *> node_traceback;
    auto node = FindLeafNode(key, &node_traceback, false);
    if (node->GetSize() == 0) {
      root_->rw_latch_.unlock();
      root_latch_.unlock();
      return End();
    }

    size_t pos = node->GetPositionToInsert(key);

    if (node == root_) root_latch_.unlock();

    if (pos >= node->GetSize()) {
      auto new_node = node->GetNextPtr();
      if (new_node != nullptr) {
        // Return retry iterator
        if (!new_node->rw_latch_.try_lock_read()) {
          node->rw_latch_.unlock();
          return IndexIterator(nullptr, 1, 1);
        }
      }
      node->rw_latch_.unlock();
      node = new_node;
      pos = 0;
    }

    return IndexIterator(node, pos, 0);
  }

  /*
   * Returns iterator denoting the end of the B+ tree
   */
  IndexIterator End() { return IndexIterator(nullptr, 0, 0); }

  /*
   * Inputs - key
   * Returns the last iterator which has key <= key
   */
  IndexIterator End(const KeyType &key) {
    std::stack<InnerNode *> node_traceback;
    auto node = FindLeafNode(key, &node_traceback, false);
    if (node->GetSize() == 0) {
      root_->rw_latch_.unlock();
      root_latch_.unlock();
      return End();
    }

    auto pos = node->GetPositionLessThanEqualTo(key);

    // if root latch not released by FindLeafNode
    if (node == root_) root_latch_.unlock();

    // If value <= key not found in current node
    if (pos == -1) {
      auto new_node = dynamic_cast<LeafNode *>(node->GetPrevPtr());
      // rend reached
      if (new_node == nullptr) {
        node->rw_latch_.unlock();
        return End();
      }
      if (!new_node->rw_latch_.try_lock_read()) {
        node->rw_latch_.unlock();
        return IndexIterator(nullptr, 1, 1);
      }
      node->rw_latch_.unlock();
      node = new_node;
      pos = new_node->GetSize() - 1;
    }
    int val_off = (node->GetEntriesEnd() - 1)->second.size() - 1;
    return IndexIterator(node, pos, val_off);
  }

  /*
   * Inputs - key1, key2
   * Returns key1 >= key2
   */
  bool KeyCmpGreaterEqual(const KeyType &key1, const KeyType &key2) { return !KEY_CMP_OBJ(key1, key2); }

  /*
   * Returns the iterator denoting retry action
   */
  IndexIterator GetRetryIterator() { return IndexIterator(nullptr, 1, 1); }

  bool CheckStructuralIntegrityHelper(Node *node, size_t height_from_root) {
    if (node->IsLeaf()) {
      if (node != root_ && node->GetSize() < MIN_KEYS_LEAF_NODE) {
        return false;
      }

      // All leaf nodes must be at the same level
      if (height_from_root != 0) {
        return false;
      }

      auto it = dynamic_cast<LeafNode *>(node)->GetEntriesBegin();

      for (size_t i = 1; i < node->GetSize(); i++) {
        // Check order of keys
        if ((it + i - 1)->first > (it + i)->first) {
          return false;
        }

        // Ensure keys have at least one value
        if ((it + i)->second.empty()) {
          return false;
        }
      }
    } else {
      if (node != root_ && node->GetSize() < MIN_KEYS_INNER_NODE) {
        return false;
      }

      // Root must have at least one key if it is not the leaf node
      if (node == root_ && node->GetSize() == 0) {
        return false;
      }

      auto it = dynamic_cast<InnerNode *>(node)->GetEntriesBegin();

      for (size_t i = 0; i < node->GetSize(); i++) {
        if (i == 0) {
          // If there is atleast one key, prev ptr should exist
          if (node->GetPrevPtr() == nullptr || (node->GetPrevPtr()->GetLastKey() >= (it + i)->first)) {
            return false;
          }
          continue;
        }

        // Check order of keys
        if ((it + i - 1)->first > (it + i)->first) {
          return false;
        }

        // Ensures that each key has a non-NULL associated pointer
        if ((it + i)->second == nullptr) {
          return false;
        }

        // Recursive function call for all children of inner node
        if (!CheckStructuralIntegrityHelper((it + i)->second, height_from_root - 1)) {
          return false;
        }

        // Check the key limits of child node
        if ((it + i - 1)->first > (it + i - 1)->second->GetFirstKey() &&
            (it + i)->first <= (it + i - 1)->second->GetFirstKey()) {
          return false;
        }

        // Check the last pointer
        if (i == node->GetSize() - 1) {
          if ((it + i)->first > (it + i)->second->GetFirstKey()) {
            return false;
          }
        }
      }
    }

    return true;
  }

  /*
   * checkStructuralIntegrity: Function to check the structural integrity for the tree
   * 1) Checks the limit for number of keys in the inner and leaf nodes
   * 2) Checks the keys stored in a node are in the ascending order
   * 3) Checks that all leaf nodes are at the same level
   * 4) Checks that all pointers in the inner node are non-NULL
   * 5) Checks that the keys in a child node are within the range specified by the parent nodes
   */
  bool CheckStructuralIntegrity() {
    if (root_->GetSize() != 0) {
      return CheckStructuralIntegrityHelper(root_, GetHeightOfTree() - 1);
    }
    return true;
  }
};
}  // namespace noisepage::storage::index