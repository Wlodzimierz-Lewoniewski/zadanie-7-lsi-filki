import numpy as np
import regex

def clean_text(text):
    return regex.sub(r'\p{P}+', '', text).lower().split()

def main():
    input_data = []
    while True:
        try:
            line = input()
            if line.strip():
                input_data.append(line.strip())
            else:
                break
        except EOFError:
            break

    n = int(input_data[0])
    documents = [clean_text(doc) for doc in input_data[1:n+1]]  
    query = clean_text(input_data[n+1])  
    k = int(input_data[n+2])

    terms = sorted(list(set(word for doc in documents for word in doc))) 
    term_index = {term: idx for idx, term in enumerate(terms)}

    term_doc_matrix = np.zeros((len(terms), n))
    for doc_id, doc in enumerate(documents):
        for word in doc:
            if word in term_index:
                term_doc_matrix[term_index[word], doc_id] = 1

    #print("\nMacierz term dokument:")
    print(term_doc_matrix)

    U, S, Vt = np.linalg.svd(term_doc_matrix, full_matrices=False)
    
    #print("\nMacierz U (Lewa Macierz Osobliwa):")
    #print(U)
    #print("\nWartości Osobliwe (Diagonalna macierzy Σ):")
    #print(S)
    #print("\nMacierz V^T (Prawa Macierz Osobliwa):")
    #print(Vt)
    
    Uk = U[:, :k]
    Sk = np.diag(S[:k])  # Macierz diagonalna z top k singular values
    Vk = Vt[:k, :]       # Pierwsze k wierszy V^T

    Ck = np.dot(Uk, np.dot(Sk, Vk))
    #print("\nMacierz Przybliżona Rzędu k (Ck)::")
    #print(Ck)
    Ck = np.dot(Sk, Vk)
    #print("kształt Ck:", Ck.shape) # Debugowanie kształtu macierzy

    query_vec = np.zeros(len(terms))
    for word in query:
        if word in term_index:
            query_vec[term_index[word]] += 1

    #print("\nQuery wektor:")
    #print(query_vec)

   # reduced_query = np.dot(np.linalg.inv(Sk), np.dot(Uk.T, query_vec))
   #print("\nZredukowany Query wektor:")
    #print(reduced_query)

    def cosine_similarity(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    similarities = []
    for doc_vec in np.dot(Sk, Vk).T:
        similarity = cosine_similarity(reduced_query, doc_vec)
        similarities.append(round(similarity, 2))

    print("\nPodobienstwa:")
    print(similarities)
    return similarities
if __name__ == "__main__":
    main()
